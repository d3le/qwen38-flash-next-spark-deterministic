#!/bin/bash
# Compare the running server's outputs with the reference (bitwise).
# Extra arguments go to tools/verify.py (--all-lengths, --skip-262k).
. "$(dirname "$0")/common.sh"
OUT=$QWEN38_ROOT/results/verify-$(date +%Y%m%d-%H%M%S)
python3 "$REPO/tools/verify.py" --url "$URL" --out "$OUT" --steps "$SPEC_STEPS" "$@"
