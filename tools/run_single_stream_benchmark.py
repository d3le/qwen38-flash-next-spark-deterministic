#!/usr/bin/env python3
"""Run one streaming SGLang request and save reproducible benchmark metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


PLE_COUNTERS = (
    "requested_rows",
    "cache_hits",
    "cache_misses",
    "backing_read_calls",
    "backing_bytes_read",
    "backing_read_time_ns",
    "lookup_calls",
    "lookup_time_ns",
)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def memory_snapshot(elapsed_s: float) -> dict[str, float | int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    vmstat = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in {"pswpin", "pswpout"}:
            vmstat[key] = int(value)
    return {
        "elapsed_s": elapsed_s,
        "physical_used_gib": (values["MemTotal"] - values["MemAvailable"]) / 2**30,
        "mem_available_gib": values["MemAvailable"] / 2**30,
        "swap_used_gib": (values["SwapTotal"] - values["SwapFree"]) / 2**30,
        "pswpin_pages": vmstat.get("pswpin", 0),
        "pswpout_pages": vmstat.get("pswpout", 0),
    }


def summarize_memory(
    samples: list[dict[str, float | int]], first_event_s: float | None
) -> dict[str, Any]:
    if not samples:
        return {}
    page_mib = os.sysconf("SC_PAGE_SIZE") / 2**20
    rates = []
    simultaneous = 0
    for before, after in zip(samples, samples[1:]):
        dt = float(after["elapsed_s"]) - float(before["elapsed_s"])
        if dt <= 0:
            continue
        swapin_pages = int(after["pswpin_pages"]) - int(before["pswpin_pages"])
        swapout_pages = int(after["pswpout_pages"]) - int(before["pswpout_pages"])
        if swapin_pages > 0 and swapout_pages > 0:
            simultaneous += 1
        rates.append((swapin_pages * page_mib / dt, swapout_pages * page_mib / dt))
    decode_samples = (
        [x for x in samples if float(x["elapsed_s"]) >= first_event_s]
        if first_event_s is not None
        else []
    )
    decode_slope = None
    if len(decode_samples) >= 2:
        x0 = float(decode_samples[0]["elapsed_s"])
        xs = [float(x["elapsed_s"]) - x0 for x in decode_samples]
        ys = [float(x["physical_used_gib"]) for x in decode_samples]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator:
            decode_slope = (
                sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
                / denominator
                * 60
            )
    first = samples[0]
    last = samples[-1]
    return {
        "sample_count": len(samples),
        "sample_interval_observed_s_mean": (
            (float(last["elapsed_s"]) - float(first["elapsed_s"])) / (len(samples) - 1)
            if len(samples) > 1
            else None
        ),
        "peak_physical_memory_gib": max(float(x["physical_used_gib"]) for x in samples),
        "minimum_mem_available_gib": min(float(x["mem_available_gib"]) for x in samples),
        "swap_used_gib_start": float(first["swap_used_gib"]),
        "swap_used_gib_end": float(last["swap_used_gib"]),
        "swapin_mib_delta": (int(last["pswpin_pages"]) - int(first["pswpin_pages"])) * page_mib,
        "swapout_mib_delta": (int(last["pswpout_pages"]) - int(first["pswpout_pages"])) * page_mib,
        "max_swapin_mib_s": max((x[0] for x in rates), default=0.0),
        "max_swapout_mib_s": max((x[1] for x in rates), default=0.0),
        "simultaneous_swapin_swapout_intervals": simultaneous,
        "decode_sample_count": len(decode_samples),
        "decode_physical_gib_start": (
            float(decode_samples[0]["physical_used_gib"]) if decode_samples else None
        ),
        "decode_physical_gib_end": (
            float(decode_samples[-1]["physical_used_gib"]) if decode_samples else None
        ),
        "decode_physical_gib_peak": (
            max(float(x["physical_used_gib"]) for x in decode_samples)
            if decode_samples
            else None
        ),
        "decode_physical_growth_gib_per_minute_linear_fit": decode_slope,
    }


def ple_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in PLE_COUNTERS
    }
    requested = result["requested_rows"]
    result["cache_hit_ratio"] = (
        result["cache_hits"] / requested if requested else None
    )
    result["backing_read_mib"] = result["backing_bytes_read"] / 2**20
    result["backing_read_time_s"] = result["backing_read_time_ns"] / 1e9
    result["lookup_time_s"] = result["lookup_time_ns"] / 1e9
    return result


def ple_stage_deltas(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    before_stages = before.get("stages", {})
    after_stages = after.get("stages", {})
    return {
        stage: ple_delta(before_stages.get(stage, {}), stage_stats)
        for stage, stage_stats in sorted(after_stages.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000/generate")
    request_input = parser.add_mutually_exclusive_group(required=True)
    request_input.add_argument("--prompt")
    request_input.add_argument("--input-ids-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--ple-stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample-interval", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_ids = None
    input_sha256 = None
    if args.input_ids_file is not None:
        input_bytes = args.input_ids_file.read_bytes()
        input_ids = json.loads(input_bytes)
        if not isinstance(input_ids, list) or not all(
            isinstance(token_id, int) for token_id in input_ids
        ):
            raise ValueError("--input-ids-file must contain one JSON integer list")
        input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    before = read_json(args.ple_stats)
    memory_samples: list[dict[str, float | int]] = []
    stop = threading.Event()
    started = time.perf_counter()

    def monitor() -> None:
        while not stop.is_set():
            memory_samples.append(memory_snapshot(time.perf_counter() - started))
            stop.wait(args.sample_interval)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    request_body = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
        },
        "stream": True,
    }
    if args.sampling_seed is not None:
        request_body["sampling_params"]["sampling_seed"] = args.sampling_seed
    if input_ids is None:
        request_body["text"] = args.prompt
    else:
        request_body["input_ids"] = input_ids
    payload = json.dumps(request_body).encode()
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    first_event_s: float | None = None
    events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(request) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                event = json.loads(line)
                if first_event_s is None:
                    first_event_s = time.perf_counter() - started
                events.append(event)
    finally:
        elapsed_s = time.perf_counter() - started
        stop.set()
        monitor_thread.join()

    after = read_json(args.ple_stats)
    final_event = events[-1] if events else {}
    meta = final_event.get("meta_info", {})
    completion_tokens = int(meta.get("completion_tokens", 0))
    prompt_tokens = int(meta.get("prompt_tokens", 0))
    decode_s = (
        elapsed_s - first_event_s
        if first_event_s is not None and elapsed_s > first_event_s
        else None
    )
    delta = ple_delta(before, after)
    stage_deltas = ple_stage_deltas(before, after)
    for stage_delta in stage_deltas.values():
        stage_delta["lookup_ms_per_output_token"] = (
            stage_delta["lookup_time_s"] * 1000 / completion_tokens
            if completion_tokens
            else None
        )
        stage_delta["backing_bytes_per_output_token"] = (
            stage_delta["backing_bytes_read"] / completion_tokens
            if completion_tokens
            else None
        )
    result = {
        "label": args.label,
        "prompt": args.prompt,
        "input_ids_file": (
            str(args.input_ids_file) if args.input_ids_file is not None else None
        ),
        "input_tokens_submitted": len(input_ids) if input_ids is not None else None,
        "input_sha256": input_sha256,
        "max_new_tokens": args.max_new_tokens,
        "sampling_seed": args.sampling_seed,
        "event_count": len(events),
        "ttft_s": first_event_s,
        "elapsed_s": elapsed_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tok_s": completion_tokens / elapsed_s if elapsed_s else None,
        "ms_per_output_token": (
            elapsed_s * 1000 / completion_tokens if completion_tokens else None
        ),
        "decode_after_first_event_s": decode_s,
        "decode_after_first_event_tok_s": (
            (completion_tokens - 1) / decode_s
            if decode_s and completion_tokens > 1
            else None
        ),
        "prefill_tok_s_approx": (
            prompt_tokens / first_event_s if first_event_s and prompt_tokens else None
        ),
        "peak_physical_memory_gib": (
            max((float(x["physical_used_gib"]) for x in memory_samples), default=None)
        ),
        "min_physical_memory_gib": (
            min((float(x["physical_used_gib"]) for x in memory_samples), default=None)
        ),
        "memory_stability": summarize_memory(memory_samples, first_event_s),
        "memory_samples": memory_samples,
        "ple_ms_per_output_token": (
            delta["lookup_time_s"] * 1000 / completion_tokens
            if completion_tokens
            else None
        ),
        "ple_delta": delta,
        "ple_stage_deltas": stage_deltas,
        "meta_info": meta,
        "text": final_event.get("text", ""),
        "output_sha256": hashlib.sha256(
            final_event.get("text", "").encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    printable = dict(result)
    printable.pop("memory_samples", None)
    printable["memory_samples_file"] = str(args.output)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
