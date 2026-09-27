#!/bin/bash
# Download nvidia/Qwen3.8-Flash-Next-NVFP4 at the pinned revision (124 GiB)
# and verify every weight file against the Hugging Face sha256.
. "$(dirname "$0")/common.sh"

mkdir -p "$BASE_DIR"
log "downloading $HF_REPO@${HF_REVISION:0:8} into $BASE_DIR (resumable)"
docker run --rm -u "$(id -u):$(id -g)" -e HF_TOKEN -e HF_HOME=/tmp/hf \
  -v "$BASE_DIR":/out "$IMAGE" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', revision='$HF_REVISION', local_dir='/out')
"
log "verifying sha256 (takes a few minutes)"
(cd "$BASE_DIR" && sha256sum -c "$REPO/data/nvidia-base.sha256")
log "base checkpoint OK"
