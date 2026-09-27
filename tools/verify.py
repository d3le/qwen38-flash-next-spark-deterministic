#!/usr/bin/env python3
"""Check a running server against the reference outputs (greedy, bitwise).

The serving stack is deterministic, so the same image, model files and
settings must reproduce the reference token ids exactly.  Checks:

1. C1: the 4 fixed benchmark prompts, 128 tokens each (output sha256).
2. Long prompts: prefixes of the 262K surrogate input (4K-33K tokens).
3. 262K: one cold prefill and one warm resend that must hit the prefix
   cache (259,968 cached tokens) and still give the same output.

Output hashes do not depend on --speculative-num-steps (3 or 4); the
acceptance length and verify count are compared only for the reference
setting (3 steps).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUICK_LENGTHS = (4096, 16385, 32896)


def flush(url: str) -> None:
    request = urllib.request.Request(
        f"{url}/flush_cache", data=b"{}", headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(request, timeout=60).read()


def single(url: str, ids_file: Path, max_new: int, out: Path, label: str) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "tools/run_single_stream_benchmark.py"),
         "--url", f"{url}/generate", "--input-ids-file", str(ids_file),
         "--max-new-tokens", str(max_new), "--output", str(out), "--label", label],
        check=True, stdout=subprocess.DEVNULL,
    )
    return json.loads(out.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3, help="--speculative-num-steps of the server")
    parser.add_argument("--all-lengths", action="store_true", help="all 12 long prefixes instead of 3")
    parser.add_argument("--skip-262k", action="store_true")
    args = parser.parse_args()

    ref = json.loads((ROOT / "data/reference.json").read_text())
    exact = args.steps == ref["server"]["speculative_num_steps"]
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, bool, str]] = []

    def check(name, ok, detail):
        rows.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    # 1. C1 fixed prompts
    c1_dir = args.out / "c1"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/run_fixed_concurrent_benchmark.py"),
         "--base-url", args.url, "--output-dir", str(c1_dir), "--concurrency", "1",
         "--runs", "2", "--max-new-tokens", "128", "--warmup"],
        check=True, stdout=subprocess.DEVNULL,
    )
    summary = json.loads((c1_dir / "summary.json").read_text())["summary"]
    for prompt, want in ref["c1"]["output_sha256"].items():
        got = sorted(set(summary["by_prompt"][prompt]["output_hashes"]))
        check(f"c1/{prompt}", got == [want], f"{got[0][:12] if got else '-'} (want {want[:12]})")
    if exact:
        accept = summary["spec_accept_length"]["median"]
        check("c1/accept", abs(accept - ref["c1"]["accept_length"]) < 1e-3,
              f"{accept:.4f} (want {ref['c1']['accept_length']:.4f})")

    # 2. long prefixes
    source = json.loads((ROOT / "workloads/mtp-262k-surrogate-input-ids.json").read_text())
    cases = ref["long_prefixes"]["cases"]
    lengths = [int(k) for k in cases] if args.all_lengths else list(QUICK_LENGTHS)
    for length in lengths:
        ids_file = args.out / f"prefix-{length}.json"
        ids_file.write_text(json.dumps(source[:length]))
        flush(args.url)
        result = single(args.url, ids_file, ref["long_prefixes"]["max_new_tokens"],
                        args.out / f"long-{length}.json", f"L{length}")
        want = cases[str(length)]
        got = result["output_sha256"]
        ok = got == want["output_sha256"]
        detail = f"{got[:12]} (want {want['output_sha256'][:12]})"
        if exact:
            meta = result.get("meta_info", {})
            ok &= meta.get("spec_verify_ct") == want["verify_ct"]
            detail += f", verify {meta.get('spec_verify_ct')} (want {want['verify_ct']})"
        check(f"long/{length}", ok, detail)

    # 3. 262K cold + warm
    if not args.skip_262k:
        ids_file = ROOT / "workloads/mtp-262k-surrogate-input-ids.json"
        want = ref["long_262k"]
        flush(args.url)
        cold = single(args.url, ids_file, want["max_new_tokens"], args.out / "262k-cold.json", "262k-cold")
        warm = single(args.url, ids_file, want["max_new_tokens"], args.out / "262k-warm.json", "262k-warm")
        check("262k/cold", cold["output_sha256"] == want["cold"]["output_sha256"],
              f"{cold['output_sha256'][:12]}, TTFT {cold.get('ttft_s', 0):.1f}s")
        cached = warm.get("meta_info", {}).get("cached_tokens")
        check("262k/warm", warm["output_sha256"] == want["warm"]["output_sha256"]
              and cached == want["warm"]["cached_tokens"],
              f"{warm['output_sha256'][:12]}, cached {cached}, TTFT {warm.get('ttft_s', 0):.2f}s")

    failed = [name for name, ok, _ in rows if not ok]
    (args.out / "verify.json").write_text(json.dumps(
        {"passed": not failed, "failed": failed, "checks": rows, "steps": args.steps}, indent=1))
    print(f"\n{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}"
          f"  ({len(rows) - len(failed)}/{len(rows)}; results in {args.out})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
