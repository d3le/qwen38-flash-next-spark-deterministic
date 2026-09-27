#!/bin/bash
# Build the served model from the NVIDIA checkpoint with the shipped
# calibration: C1 rewrites the LM head to FP8, C2 adds FP8 GDN projections
# (36 of 48 layers).  Only the changed shards are written (~8 GB); everything
# else is a symlink to the base checkpoint.  The result is checked against
# data/converted.sha256, so a mismatch means a different model.
. "$(dirname "$0")/common.sh"

for dir in "$C1_DIR" "$C2_DIR"; do
  if [ -e "$dir" ]; then
    echo "$dir already exists; remove it to convert again" >&2
    exit 1
  fi
done
mkdir -p "$QWEN38_ROOT"
log "converting (C1: LM head FP8, C2: GDN projections FP8)"
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$BASE_DIR":/base-model:ro -v "$REPO":/recipe:ro -v "$QWEN38_ROOT":/work \
  "$IMAGE" bash -c "
set -e
python3 /recipe/tools/convert_nvidia_lm_head_fp8.py --source /base-model \
  --calibration /recipe/data/calibration/c1-lm-head-input.json \
  --output /work/c1 --source-mount-target /base-model > /work/c1-conversion.json
python3 /recipe/tools/convert_nvidia_c2_gdn_fp8.py --source /work/c1 \
  --calibration /recipe/data/calibration/c2-gdn.json \
  --output /work/c2 --source-mount-target /c1-model > /work/c2-conversion.json
"
log "verifying the converted files"
(cd "$QWEN38_ROOT" && sha256sum -c "$REPO/data/converted.sha256")
log "model ready in $C2_DIR"
