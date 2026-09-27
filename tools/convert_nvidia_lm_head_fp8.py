#!/usr/bin/env python3
"""Build a non-destructive NVIDIA-C1 LM-head-FP8 replacement-shard model.

The NVIDIA checkpoint is left untouched.  All unchanged files in the output
directory are symlinks to ``source_mount_target``; the shard containing
``lm_head.weight`` is rewritten once with an FP8 weight and the two ModelOpt
scale tensors.  This avoids duplicate tensors and therefore does not require a
custom SGLang safetensors-index loader patch.

The output is intended to be mounted at ``/model`` together with the original
checkpoint mounted read-only at the path passed through ``--source-mount-target``.
Generated safetensors files are runtime artifacts, not Git objects.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

FP8_MAX = 448.0
METADATA_FILES = {
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_exact(values: Any, item: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"expected a list containing {item!r}, got {type(values).__name__}")
    return [value for value in values if value != item]


def _set_lm_head_fp8(quant: dict[str, Any], *, flat: bool) -> dict[str, Any]:
    """Add the LM-head FP8 policy to either ModelOpt config representation."""
    result = copy.deepcopy(quant)
    if flat:
        if str(result.get("quant_algo", "")).upper() != "MIXED_PRECISION":
            raise ValueError("config.json quantization_config is not MIXED_PRECISION")
        result["ignore"] = _remove_exact(result.get("ignore", []), "lm_head")
        layers = dict(result.get("quantized_layers", {}))
        existing = layers.get("lm_head")
        if existing is not None and str(existing.get("quant_algo", "")).upper() != "FP8":
            raise ValueError(f"lm_head already has incompatible policy: {existing!r}")
        layers["lm_head"] = {"quant_algo": "FP8"}
        result["quantized_layers"] = layers
    else:
        nested = result.get("quantization")
        if not isinstance(nested, dict):
            raise ValueError("hf_quant_config.json lacks a quantization section")
        if str(nested.get("quant_algo", "")).upper() != "MIXED_PRECISION":
            raise ValueError("hf_quant_config.json quantization is not MIXED_PRECISION")
        nested["exclude_modules"] = _remove_exact(
            nested.get("exclude_modules", []), "lm_head"
        )
        layers = dict(nested.get("quantized_layers", {}))
        existing = layers.get("lm_head")
        if existing is not None and str(existing.get("quant_algo", "")).upper() != "FP8":
            raise ValueError(f"lm_head already has incompatible policy: {existing!r}")
        layers["lm_head"] = {"quant_algo": "FP8"}
        nested["quantized_layers"] = layers
    return result


def input_scale_from_calibration(path: Path) -> float:
    payload = json.loads(path.read_text())
    values = payload.get("checkpoint_input_scales", {})
    value = values.get("lm_head.input_scale")
    if value is None:
        value = payload.get("input_scale")
    if value is None:
        value = payload.get("lm_head", {}).get("input_scale")
    if value is None:
        raise ValueError(
            f"{path} must contain checkpoint_input_scales.lm_head.input_scale "
            "or input_scale"
        )
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"invalid LM-head input scale: {value}")
    return value


def quantize_weight(weight):
    import torch

    if weight.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 lm_head.weight, got {weight.dtype}")
    amax = float(weight.detach().float().abs().amax().item())
    scale = max(amax / FP8_MAX, 1e-12)
    qweight = (
        (weight.float() / scale)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return qweight, torch.tensor(scale, dtype=torch.float32)


def convert(
    source: Path,
    calibration: Path,
    output: Path,
    source_mount_target: str = "/base-model",
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source = source.resolve()
    config_path = source / "config.json"
    hf_quant_path = source / "hf_quant_config.json"
    index_path = source / "model.safetensors.index.json"
    for required in (config_path, hf_quant_path, index_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = json.loads(config_path.read_text())
    hf_quant = json.loads(hf_quant_path.read_text())
    index = json.loads(index_path.read_text())
    weight_map = dict(index.get("weight_map", {}))
    source_shard = weight_map.get("lm_head.weight")
    if not source_shard:
        raise ValueError("model index has no lm_head.weight")
    if any(name in weight_map for name in ("lm_head.weight_scale", "lm_head.input_scale")):
        raise ValueError("source already contains LM-head FP8 scale tensors")
    input_scale = input_scale_from_calibration(calibration)

    original_shard = source / source_shard
    with safe_open(original_shard, framework="pt", device="cpu") as handle:
        if "lm_head.weight" not in handle.keys():
            raise ValueError(f"{original_shard} does not contain lm_head.weight")
        original_weight = handle.get_tensor("lm_head.weight")
        qweight, weight_scale = quantize_weight(original_weight)
        original_bytes = original_weight.numel() * original_weight.element_size()
        quantized_bytes = qweight.numel() * qweight.element_size()
        replacement_tensors = {
            name: handle.get_tensor(name)
            for name in handle.keys()
            if name != "lm_head.weight"
        }
    replacement_tensors.update(
        {
            "lm_head.weight": qweight,
            "lm_head.weight_scale": weight_scale,
            "lm_head.input_scale": torch.tensor(input_scale, dtype=torch.float32),
        }
    )

    mixed = _set_lm_head_fp8(config["quantization_config"], flat=True)
    config["quantization_config"] = mixed
    if isinstance(config.get("text_config"), dict) and "quantization_config" in config["text_config"]:
        config["text_config"]["quantization_config"] = copy.deepcopy(mixed)
    hf_mixed = _set_lm_head_fp8(hf_quant, flat=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        for entry in source.iterdir():
            if not entry.is_file() or entry.name in METADATA_FILES:
                continue
            destination = staging / entry.name
            if entry.name == source_shard:
                save_file(replacement_tensors, destination, metadata={"format": "pt"})
            else:
                os.symlink(f"{source_mount_target.rstrip('/')}/{entry.name}", destination)

        weight_map["lm_head.weight_scale"] = source_shard
        weight_map["lm_head.input_scale"] = source_shard
        new_index = copy.deepcopy(index)
        new_index["weight_map"] = weight_map
        metadata = dict(new_index.get("metadata", {}))
        if isinstance(metadata.get("total_size"), int):
            metadata["total_size"] = (
                metadata["total_size"]
                - original_bytes
                + quantized_bytes
                + 8
            )
        new_index["metadata"] = metadata
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "hf_quant_config.json").write_text(
            json.dumps(hf_mixed, indent=2) + "\n"
        )
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": "qwen38-nvidia-c1-lm-head-fp8-replacement-v1",
            "source": str(source),
            "output": str(output),
            "source_mount_target": source_mount_target,
            "source_config_sha256": sha256(config_path),
            "source_hf_quant_config_sha256": sha256(hf_quant_path),
            "source_index_sha256": sha256(index_path),
            "source_shard": source_shard,
            "source_shard_sha256": sha256(original_shard),
            "replacement_shard_sha256": sha256(staging / source_shard),
            "input_scale": input_scale,
            "weight_scale": float(weight_scale.item()),
            "original_lm_head_bytes": original_bytes,
            "quantized_lm_head_bytes": quantized_bytes,
            "replacement_shard_bytes": (staging / source_shard).stat().st_size,
            "precision": {
                "lm_head": "FP8 E4M3 static per-tensor weight + calibrated input scale",
                "base": "NVIDIA NVFP4 checkpoint unchanged except LM-head policy",
                "gdn_qsa_mtp_ple_kv": "unchanged",
            },
            "loader_patch_required": False,
            "calibration_sha256": sha256(calibration),
        }
        (staging / "nvidia-c1-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.rename(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-mount-target", default="/base-model")
    args = parser.parse_args()
    print(
        json.dumps(
            convert(args.source, args.calibration, args.output, args.source_mount_target),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
