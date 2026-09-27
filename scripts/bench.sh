#!/bin/bash
# Single-stream decode speed: C1 and long prompts.  Run right after start.sh.
. "$(dirname "$0")/common.sh"
OUT=$QWEN38_ROOT/results/bench-$(date +%Y%m%d-%H%M%S)
python3 "$REPO/tools/bench.py" --url "$URL" --out "$OUT" "$@"
