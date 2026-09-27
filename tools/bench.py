#!/usr/bin/env python3
"""Single-stream decode benchmark: C1 (4 fixed prompts x 128 tokens) and
long prompts (prefixes of the 262K surrogate input).  Measure right after a
start: long requests first can shift C1 by up to ~2 ms/verify."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def flush(url: str) -> None:
    request = urllib.request.Request(
        f"{url}/flush_cache", data=b"{}", headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(request, timeout=60).read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--lengths", default="4096,8192,16384,32896")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, str(ROOT / "tools/run_fixed_concurrent_benchmark.py"),
         "--base-url", args.url, "--output-dir", str(args.out / "c1"), "--concurrency", "1",
         "--runs", str(args.runs), "--max-new-tokens", "128", "--warmup"],
        check=True, stdout=subprocess.DEVNULL,
    )
    s = json.loads((args.out / "c1" / "summary.json").read_text())["summary"]
    report = {"c1": {
        "decode_tok_s": s["decode_tok_s"]["median"],
        "ms_per_verify": s["ms_per_verify"]["median"],
        "accept_length": s["spec_accept_length"]["median"],
    }}
    print(f"C1: {report['c1']['decode_tok_s']:.2f} tok/s, {report['c1']['ms_per_verify']:.2f} ms/verify, "
          f"accept {report['c1']['accept_length']:.3f}", flush=True)

    source = json.loads((ROOT / "workloads/mtp-262k-surrogate-input-ids.json").read_text())
    long = {}
    for length in [int(v) for v in args.lengths.split(",")]:
        ids_file = args.out / f"prefix-{length}.json"
        ids_file.write_text(json.dumps(source[:length]))
        flush(args.url)
        out = args.out / f"long-{length}.json"
        subprocess.run(
            [sys.executable, str(ROOT / "tools/run_single_stream_benchmark.py"),
             "--url", f"{args.url}/generate", "--input-ids-file", str(ids_file),
             "--max-new-tokens", "128", "--output", str(out), "--label", f"L{length}"],
            check=True, stdout=subprocess.DEVNULL,
        )
        x = json.loads(out.read_text())
        meta = x.get("meta_info", {})
        verify = meta.get("spec_verify_ct") or 0
        row = {
            "decode_tok_s": x["decode_after_first_event_tok_s"],
            "ms_per_verify": 1000 * x["decode_after_first_event_s"] / max(verify - 1, 1),
            "accept_length": meta.get("spec_accept_length"),
            "ttft_s": x["ttft_s"],
        }
        long[length] = row
        print(f"L={length}: {row['decode_tok_s']:.2f} tok/s, {row['ms_per_verify']:.2f} ms/verify, "
              f"accept {row['accept_length']:.3f}, TTFT {row['ttft_s']:.2f}s", flush=True)
    report["long"] = long
    report["long_mean_tok_s"] = statistics.mean(r["decode_tok_s"] for r in long.values())
    print(f"long mean: {report['long_mean_tok_s']:.2f} tok/s")
    (args.out / "bench.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
