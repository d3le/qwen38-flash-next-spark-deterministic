#!/usr/bin/env python3
"""Build an NVIDIA-C2 GDN-FP8 overlay on top of an NVIDIA-C1 model.

The input is the immutable C1 candidate (LM head FP8).  Only the selected
GDN projection tensors are rewritten into small sidecar safetensors shards;
all other files are symlinked to the C1 mount.  The existing ModelOpt mixed
precision policy is preserved and extended, so C1's LM-head policy remains
active.  The source model and C1 candidate are never modified.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

FP8_MAX = 448.0
NUM_LAYERS = 48
LINEAR_LAYERS = [i for i in range(NUM_LAYERS) if (i + 1) % 4 != 0]
PROJECTIONS = ("in_proj_qkv", "in_proj_z", "out_proj")
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


def selected_weight_names() -> list[str]:
    return [
        f"model.language_model.layers.{layer}.linear_attn.{projection}.weight"
        for layer in LINEAR_LAYERS
        for projection in PROJECTIONS
    ]


def _selected_prefixes() -> list[str]:
    return [
        f"model.language_model.layers.{layer}.linear_attn"
        for layer in LINEAR_LAYERS
    ]


def _remove_selected_ignore(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"expected a list, got {type(values).__name__}")
    prefixes = _selected_prefixes()
    result = []
    for pattern in values:
        # Remove only concrete per-layer linear-attention exclusions.  A
        # broad wildcard is retained so the protected layers remain BF16.
        if any(
            fnmatch.fnmatch(prefix + ".probe", str(pattern))
            and not any(token in str(pattern) for token in ("*", "?"))
            for prefix in prefixes
        ):
            continue
        if any(
            str(pattern) in (prefix + "*", prefix + ".*") for prefix in prefixes
        ):
            continue
        result.append(pattern)
    return result


def _extend_mixed_policy(policy: dict[str, Any], *, nested: bool) -> dict[str, Any]:
    result = copy.deepcopy(policy)
    if nested:
        layers = dict(result.get("quantized_layers", {}))
        result["exclude_modules"] = _remove_selected_ignore(
            result.get("exclude_modules", [])
        )
    else:
        layers = dict(result.get("quantized_layers", {}))
        result["ignore"] = _remove_selected_ignore(result.get("ignore", []))

    if str(result.get("quant_algo", "")).upper() != "MIXED_PRECISION":
        raise ValueError("C2 source must use ModelOpt MIXED_PRECISION")
    for name in selected_weight_names():
        existing = layers.get(name.removesuffix(".weight"))
        if existing is not None and str(existing.get("quant_algo", "")).upper() != "FP8":
            raise ValueError(f"incompatible existing GDN policy for {name}: {existing}")
        layers[name.removesuffix(".weight")] = {"quant_algo": "FP8"}
    result["quantized_layers"] = layers
    return result


def calibration_values(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    values = payload.get("checkpoint_input_scales")
    if not isinstance(values, dict):
        raise ValueError("calibration must contain checkpoint_input_scales")
    expected = {
        name.removesuffix(".weight") + ".input_scale"
        for name in selected_weight_names()
    }
    missing = sorted(expected - values.keys())
    extra = sorted(set(values) - expected)
    if missing or extra:
        raise ValueError(f"invalid calibration coverage: missing={missing} extra={extra}")
    result = {}
    for name in sorted(expected):
        value = float(values[name])
        if not 0.0 < value < 10.0:
            raise ValueError(f"invalid input scale for {name}: {value}")
        result[name] = value
    return result


def _qkv_split_sizes(weight, config: dict[str, Any]) -> list[int]:
    text = config.get("text_config", config)
    key = int(text["linear_key_head_dim"]) * int(text["linear_num_key_heads"])
    value = int(text["linear_value_head_dim"]) * int(text["linear_num_value_heads"])
    sizes = [key, key, value]
    if sum(sizes) != weight.shape[0]:
        raise ValueError(f"unexpected GDN qkv shape {tuple(weight.shape)}; split={sizes}")
    return sizes


def quantize_weight(name: str, weight, config: dict[str, Any]):
    import torch

    if weight.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 {name}, got {weight.dtype}")
    split_sizes = _qkv_split_sizes(weight, config) if name.endswith("in_proj_qkv.weight") else [weight.shape[0]]
    qchunks = []
    scales = []
    start = 0
    for width in split_sizes:
        chunk = weight[start : start + width].float()
        scale = max(float(chunk.abs().amax().item()) / FP8_MAX, 1e-12)
        qchunks.append(
            (chunk / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )
        scales.append(scale)
        start += width
    return torch.cat(qchunks, dim=0).contiguous(), torch.tensor(scales, dtype=torch.float32)


def convert(
    source: Path,
    calibration: Path,
    output: Path,
    source_mount_target: str = "/c1-model",
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
    selected = selected_weight_names()
    missing = sorted(set(selected) - weight_map.keys())
    if missing:
        raise ValueError(f"C1 index is missing GDN weights: {missing[:5]}")
    existing_scales = [
        name.removesuffix(".weight") + suffix
        for name in selected
        for suffix in (".weight_scale", ".input_scale")
        if name.removesuffix(".weight") + suffix in weight_map
    ]
    if existing_scales:
        raise ValueError(f"C1 already contains selected GDN scales: {existing_scales[:5]}")
    scales = calibration_values(calibration)

    config_policy = _extend_mixed_policy(config["quantization_config"], nested=False)
    config["quantization_config"] = config_policy
    if isinstance(config.get("text_config"), dict) and "quantization_config" in config["text_config"]:
        config["text_config"]["quantization_config"] = copy.deepcopy(config_policy)
    hf_policy = _extend_mixed_policy(hf_quant["quantization"], nested=True)
    hf_quant["quantization"] = hf_policy

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        for entry in source.iterdir():
            if not entry.is_file() or entry.name in METADATA_FILES:
                continue
            os.symlink(f"{source_mount_target.rstrip('/')}/{entry.name}", staging / entry.name)

        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in selected:
            by_shard[weight_map[name]].append(name)
        overlay_records = []
        selected_original_bytes = 0
        selected_fp8_bytes = 0
        scale_scalar_count = 0

        for ordinal, (source_shard, names) in enumerate(sorted(by_shard.items()), 1):
            overlay_name = f"nvidia-c2-gdn-fp8-{ordinal:05d}-of-{len(by_shard):05d}.safetensors"
            tensors = {}
            with safe_open(source / source_shard, framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    weight = handle.get_tensor(name)
                    qweight, weight_scale = quantize_weight(name, weight, config)
                    prefix = name.removesuffix(".weight")
                    input_scale = torch.tensor(scales[prefix + ".input_scale"], dtype=torch.float32)
                    tensors[name] = qweight
                    tensors[prefix + ".weight_scale"] = weight_scale
                    tensors[prefix + ".input_scale"] = input_scale
                    selected_original_bytes += weight.numel() * weight.element_size()
                    selected_fp8_bytes += qweight.numel() * qweight.element_size()
                    scale_scalar_count += weight_scale.numel() + 1
            save_file(tensors, staging / overlay_name, metadata={"format": "pt"})
            with safe_open(staging / overlay_name, framework="pt", device="cpu") as check:
                for name in names:
                    if check.get_tensor(name).dtype != torch.float8_e4m3fn:
                        raise ValueError(f"overlay dtype validation failed: {name}")
            for name in tensors:
                weight_map[name] = overlay_name
            overlay_records.append(
                {
                    "file": overlay_name,
                    "source_shard": source_shard,
                    "tensor_count": len(tensors),
                    "bytes": (staging / overlay_name).stat().st_size,
                }
            )

        new_index = copy.deepcopy(index)
        new_index["weight_map"] = weight_map
        metadata = dict(new_index.get("metadata", {}))
        if isinstance(metadata.get("total_size"), int):
            metadata["total_size"] = (
                metadata["total_size"]
                - selected_original_bytes
                + selected_fp8_bytes
                + scale_scalar_count * 4
            )
        new_index["metadata"] = metadata
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "hf_quant_config.json").write_text(json.dumps(hf_quant, indent=2) + "\n")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": "qwen38-nvidia-c2-gdn-fp8-overlay-v1",
            "source": str(source),
            "output": str(output),
            "source_mount_target": source_mount_target,
            "source_config_sha256": sha256(config_path),
            "source_hf_quant_config_sha256": sha256(hf_quant_path),
            "source_index_sha256": sha256(index_path),
            "calibration_sha256": sha256(calibration),
            "linear_layers": LINEAR_LAYERS,
            "selected_weight_count": len(selected),
            "overlay_shards": overlay_records,
            "selected_original_bytes": selected_original_bytes,
            "selected_fp8_bytes": selected_fp8_bytes,
            "sidecar_physical_bytes": sum(item["bytes"] for item in overlay_records),
            "precision": {
                "lm_head": "C1 FP8 preserved",
                "gdn_qkv_z_out": "FP8 E4M3 static activation/per-tensor weight",
                "experts_mtp_ple_qsa_kv": "C1/base unchanged",
            },
            "loader_patch_required": False,
        }
        (staging / "nvidia-c2-manifest.json").write_text(
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
    parser.add_argument("--source-mount-target", default="/c1-model")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.calibration, args.output, args.source_mount_target), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
