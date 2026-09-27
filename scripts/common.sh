# Shared settings for the scripts in this directory.  Override any of them
# in a .env file at the repository root (see .env.sample).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
fi

QWEN38_ROOT=${QWEN38_ROOT:-$HOME/qwen38}
BASE_DIR=${BASE_DIR:-$QWEN38_ROOT/nvidia-base}
C1_DIR=$QWEN38_ROOT/c1
C2_DIR=$QWEN38_ROOT/c2
PLE_DIR=${PLE_DIR:-$QWEN38_ROOT/ple}
CACHE_DIR=${CACHE_DIR:-$QWEN38_ROOT/cache}
IMAGE=${IMAGE:-ghcr.io/d3le/qwen38-flash-next-spark-deterministic:v0.1}
CONTAINER=${CONTAINER:-qwen38}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MEM_FRACTION=${MEM_FRACTION:-0.86}
SPEC_STEPS=${SPEC_STEPS:-3}
TOKEN_MAP=${TOKEN_MAP:-$REPO/data/hot_vocab_mixed_freq_65536.pt}

HF_REPO=nvidia/Qwen3.8-Flash-Next-NVFP4
HF_REVISION=fc694b54fb0174e0913e6adf86691ef85a4ead47
URL=http://127.0.0.1:$PORT

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
