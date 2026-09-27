#!/usr/bin/env python3
"""Run a fixed-seed, fixed-length concurrent benchmark on one SGLang server.

One run is a barrier-aligned batch of up to ``--concurrency`` streaming
requests.  Every prompt is measured ``--runs`` times and the repetitions are
packed into batches, so ``--concurrency 1`` and ``--concurrency 4`` cover the
same prompt set.  Every request generates exactly ``--max-new-tokens`` tokens
(``ignore_eos``), the radix cache is flushed before each batch, and per-stream
TTFT plus decode metrics are captured from the SSE stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "qwen38-cuda-graph-fixed-seed-concurrent-v1"
SERVER_SEED = 920698591
REQUEST_SEED_BASE = 920698591
PROMPTS = [
    (
        "latency",
        "Analyze the latency of a single-stream inference server. Give a detailed "
        "10-item checklist covering measurement, prefill, decode, and speculative "
        "decoding. Control marker: titanium aurora 24680.",
    ),
    (
        "speculative",
        "Explain how speculative decoding acceptance length affects throughput. Give "
        "a detailed technical answer with mechanisms, equations in words, and "
        "concrete experiments. Control marker: cobalt quasar 13579.",
    ),
    (
        "python",
        "Write a detailed Python-oriented checklist for measuring median decode "
        "latency, token throughput, and failed requests in an inference server. "
        "Include implementation pitfalls. Control marker: amber lattice 97531.",
    ),
    (
        "japanese",
        "単一ストリーム推論の遅延要因を、測定項目、推測デコード、改善優先度を "
        "含めて詳しく説明してください。制御マーカー: sakura orbit 86420。",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30018")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3, help="repetitions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-max-new-tokens", type=int, default=32)
    parser.add_argument("--server-seed", type=int, default=SERVER_SEED)
    parser.add_argument("--request-seed-base", type=int, default=REQUEST_SEED_BASE)
    parser.add_argument("--timeout", type=float, default=1200.0)
    return parser.parse_args()


def flush_cache(base_url: str) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/flush_cache",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": True,
                "status": response.status,
                "elapsed_s": time.perf_counter() - started,
                "body": body,
            }
    except Exception as error:  # noqa: BLE001 - keep per-batch diagnostics
        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error": repr(error),
        }


def plan_batches(
    prompts: list[tuple[str, str]],
    concurrency: int,
    runs: int,
    request_seed_base: int,
) -> list[list[tuple[int, str, str, int]]]:
    """Split ``runs`` repetitions of every prompt into concurrent batches."""

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if not prompts:
        raise ValueError("prompts must not be empty")
    sequence = [index for _ in range(runs) for index in range(len(prompts))]
    batches = []
    for start in range(0, len(sequence), concurrency):
        batch = []
        for prompt_index in sequence[start:start + concurrency]:
            name, text = prompts[prompt_index]
            batch.append((prompt_index, name, text, request_seed_base + prompt_index))
        batches.append(batch)
    return batches


def stream_result(
    prompt_name: str,
    prompt: str,
    sampling_seed: int,
    max_new_tokens: int,
    ttft_s: float | None,
    finished_s: float,
    events: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    final_event = events[-1] if events else {}
    meta = final_event.get("meta_info", {})
    completion_tokens = int(meta.get("completion_tokens", 0) or 0)
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    text = str(final_event.get("text", ""))
    decode_s = (
        finished_s - ttft_s
        if ttft_s is not None and finished_s > ttft_s
        else None
    )
    verify_ct = meta.get("spec_verify_ct")
    return {
        "prompt_name": prompt_name,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "sampling_seed": sampling_seed,
        "max_new_tokens": max_new_tokens,
        "event_count": len(events),
        "ttft_s": ttft_s,
        "finished_s": finished_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tok_s": completion_tokens / finished_s if finished_s else None,
        "decode_s": decode_s,
        "decode_after_first_event_tok_s": (
            (completion_tokens - 1) / decode_s
            if decode_s is not None and completion_tokens > 1
            else None
        ),
        "ms_per_verify": (
            decode_s * 1000.0 / verify_ct
            if decode_s is not None
            and isinstance(verify_ct, (int, float))
            and verify_ct > 0
            else None
        ),
        "meta_info": meta,
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_verify_ct": verify_ct,
        "spec_correct_drafts_histogram": meta.get("spec_correct_drafts_histogram"),
        "text": text,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "error": error,
    }


def run_stream(
    base_url: str,
    prompt_name: str,
    prompt: str,
    sampling_seed: int,
    max_new_tokens: int,
    barrier: threading.Barrier,
    timeout: float,
) -> dict[str, Any]:
    barrier.wait()
    started = time.perf_counter()
    request_body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "sampling_seed": sampling_seed,
            # Every repetition must be a fixed-length measurement so EOS
            # timing cannot change the decode denominator between arms.
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    events: list[dict[str, Any]] = []
    ttft_s: float | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                event = json.loads(line)
                if ttft_s is None:
                    ttft_s = time.perf_counter() - started
                events.append(event)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        error = repr(exc)
    finished_s = time.perf_counter() - started
    return stream_result(
        prompt_name,
        prompt,
        sampling_seed,
        max_new_tokens,
        ttft_s,
        finished_s,
        events,
        error,
    )


def run_batch(
    base_url: str,
    targets: list[tuple[int, str, str, int]],
    max_new_tokens: int,
    timeout: float,
) -> list[dict[str, Any]]:
    barrier = threading.Barrier(len(targets))
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = [
            pool.submit(
                run_stream,
                base_url,
                name,
                text,
                seed,
                max_new_tokens,
                barrier,
                timeout,
            )
            for _, name, text, seed in targets
        ]
        return [future.result() for future in futures]


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [case for case in cases if case.get("error") is None]
    if not successful:
        return {
            "stream_count": len(cases),
            "successful_stream_count": 0,
            "total_completion_tokens": 0,
            "batch_wall_s": None,
            "aggregate_output_tok_s": None,
            "aggregate_decode_tok_s": None,
        }
    last_finish_s = max(float(case["finished_s"]) for case in successful)
    first_token_s = min(
        float(case["ttft_s"])
        for case in successful
        if case.get("ttft_s") is not None
    )
    total_tokens = sum(int(case["completion_tokens"]) for case in successful)
    decode_tokens = sum(
        max(int(case["completion_tokens"]) - 1, 0) for case in successful
    )
    decode_window_s = (
        last_finish_s - first_token_s if last_finish_s > first_token_s else None
    )
    return {
        "stream_count": len(cases),
        "successful_stream_count": len(successful),
        "total_completion_tokens": total_tokens,
        "batch_wall_s": last_finish_s,
        "aggregate_output_tok_s": (
            total_tokens / last_finish_s if last_finish_s else None
        ),
        "aggregate_decode_tok_s": (
            decode_tokens / decode_window_s if decode_window_s else None
        ),
    }


def metric(values: list[Any]) -> dict[str, Any]:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "values": numbers,
        "mean": statistics.mean(numbers) if numbers else None,
        "median": statistics.median(numbers) if numbers else None,
    }


def summarize(
    cases: list[dict[str, Any]], batch_aggregates: list[dict[str, Any]]
) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_prompt.setdefault(case["prompt_name"], []).append(case)
    prompt_summary = {}
    for name, prompt_cases in by_prompt.items():
        prompt_summary[name] = {
            "run_count": len(prompt_cases),
            "spec_accept_length": metric(
                [case.get("spec_accept_length") for case in prompt_cases]
            ),
            "spec_accept_rate": metric(
                [case.get("spec_accept_rate") for case in prompt_cases]
            ),
            "spec_verify_ct": metric(
                [case.get("spec_verify_ct") for case in prompt_cases]
            ),
            "completion_tokens": metric(
                [case.get("completion_tokens") for case in prompt_cases]
            ),
            "decode_tok_s": metric(
                [case.get("decode_after_first_event_tok_s") for case in prompt_cases]
            ),
            "ms_per_verify": metric(
                [case.get("ms_per_verify") for case in prompt_cases]
            ),
            "ttft_s": metric([case.get("ttft_s") for case in prompt_cases]),
            "output_hashes": [case.get("output_sha256") for case in prompt_cases],
            "errors": [case.get("error") for case in prompt_cases if case.get("error")],
        }
    return {
        "run_count": len(cases),
        "successful_run_count": sum(case.get("error") is None for case in cases),
        "spec_accept_length": metric(
            [case.get("spec_accept_length") for case in cases]
        ),
        "spec_accept_rate": metric([case.get("spec_accept_rate") for case in cases]),
        "spec_verify_ct": metric([case.get("spec_verify_ct") for case in cases]),
        "completion_tokens": metric([case.get("completion_tokens") for case in cases]),
        "decode_tok_s": metric(
            [case.get("decode_after_first_event_tok_s") for case in cases]
        ),
        "ms_per_verify": metric([case.get("ms_per_verify") for case in cases]),
        "ttft_s": metric([case.get("ttft_s") for case in cases]),
        "aggregate_output_tok_s": metric(
            [batch.get("aggregate_output_tok_s") for batch in batch_aggregates]
        ),
        "aggregate_decode_tok_s": metric(
            [batch.get("aggregate_decode_tok_s") for batch in batch_aggregates]
        ),
        "batch_wall_s": metric([batch.get("batch_wall_s") for batch in batch_aggregates]),
        "by_prompt": prompt_summary,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.max_new_tokens < 2:
        raise ValueError("max-new-tokens must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    batches = plan_batches(PROMPTS, args.concurrency, args.runs, args.request_seed_base)
    protocol: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "runs_per_prompt": args.runs,
        "batch_count": len(batches),
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
        "flush_before_every_batch": True,
        "server_seed": args.server_seed,
        "request_seed_base": args.request_seed_base,
        "prompts": [{"name": name, "text": text} for name, text in PROMPTS],
        "batch_prompts": [[name for _, name, _, _ in batch] for batch in batches],
    }
    write_json(args.output_dir / "protocol.json", protocol)

    if args.warmup:
        warmup_flush = flush_cache(args.base_url)
        warmup_cases = []
        for warmup_batch in plan_batches(
            PROMPTS, args.concurrency, 1, args.request_seed_base
        ):
            warmup_cases.extend(
                run_batch(
                    args.base_url,
                    warmup_batch,
                    args.warmup_max_new_tokens,
                    args.timeout,
                )
            )
        protocol["warmup"] = {
            "excluded": True,
            "max_new_tokens": args.warmup_max_new_tokens,
            "flush": warmup_flush,
            "aggregate": aggregate(warmup_cases),
        }
        write_json(args.output_dir / "warmup.json", protocol["warmup"])
        write_json(args.output_dir / "protocol.json", protocol)
        print(
            json.dumps({"warmup": protocol["warmup"]}, ensure_ascii=False),
            flush=True,
        )

    cases: list[dict[str, Any]] = []
    batch_aggregates: list[dict[str, Any]] = []
    protocol["batches"] = []
    for run, targets in enumerate(batches, start=1):
        flush = flush_cache(args.base_url)
        run_cases = run_batch(
            args.base_url, targets, args.max_new_tokens, args.timeout
        )
        for case in run_cases:
            case["run"] = run
        batch = {
            "run": run,
            "prompts": [name for _, name, _, _ in targets],
            "flush": flush,
            "aggregate": aggregate(run_cases),
            "cases": run_cases,
        }
        write_json(args.output_dir / f"r{run}.json", batch)
        protocol["batches"].append({"run": run, "aggregate": batch["aggregate"]})
        cases.extend(run_cases)
        batch_aggregates.append(batch["aggregate"])
        write_json(args.output_dir / "protocol.json", protocol)
        print(
            json.dumps(
                {"run": run, "flush": flush, "aggregate": batch["aggregate"]},
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "protocol": protocol,
        "summary": summarize(cases, batch_aggregates),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False), flush=True)
    if summary["summary"]["successful_run_count"] != summary["summary"]["run_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
