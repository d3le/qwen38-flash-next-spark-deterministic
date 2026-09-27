#!/bin/bash
# Start the server in the background and wait until it is healthy.
# The first start also copies the precompiled kernels into $CACHE_DIR and
# writes the 48 GiB PLE table to $PLE_DIR; later starts reuse both.
. "$(dirname "$0")/common.sh"

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container $CONTAINER already exists; run scripts/stop.sh first" >&2
  exit 1
fi
[ -f "$C2_DIR/config.json" ] || { echo "no model in $C2_DIR; run scripts/convert.sh" >&2; exit 1; }
mkdir -p "$PLE_DIR/cache/c1-prod" "$CACHE_DIR"
if [ -z "$(ls -A "$CACHE_DIR")" ]; then
  log "seeding $CACHE_DIR with the precompiled kernels from the image"
  docker run --rm -v "$CACHE_DIR":/dst "$IMAGE" cp -a /opt/qwen38/runtime-cache/. /dst/
fi

docker run -d --name "$CONTAINER" \
  --gpus all --network host --ipc host --security-opt label=disable \
  --memory 112g --memory-swap 112g \
  -e SGLANG_DETERMINISTIC_BF16_LINEAR=1 \
  -e SGLANG_FP8_POSITIONAL_PREFILL=1 \
  -e SGLANG_QWEN4_PLE_FILE_REUSE=1 \
  -e SGLANG_QWEN4_PLE_STORAGE=ssd \
  -e SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=4 \
  -e SGLANG_FLASHINFER_AUTOTUNE_CACHE=1 \
  -e SGLANG_FLASHINFER_AUTOTUNE_EXTEND=0 \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  -e SGLANG_RUST_BUILD_MODE=never \
  -e MAX_JOBS=1 -e FLASHINFER_NVCC_THREADS=1 \
  -v "$C2_DIR":/model:ro \
  -v "$C1_DIR":/c1-model:ro \
  -v "$BASE_DIR":/base-model:ro \
  -v "$PLE_DIR":/ple \
  -v "$CACHE_DIR":/root/.cache/sglang \
  -v "$TOKEN_MAP":/token-map/hot_vocab.pt:ro \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name qwen3.8-flash-next \
    --host "$HOST" --port "$PORT" --tp-size 1 \
    --context-length 262144 --max-running-requests 4 \
    --max-total-tokens 262144 --max-prefill-tokens 262144 \
    --chunked-prefill-size 8192 --mem-fraction-static "$MEM_FRACTION" \
    --random-seed 920698591 --skip-server-warmup \
    --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cutlass \
    --bf16-gemm-backend auto --kv-cache-dtype bf16 \
    --speculative-algorithm NEXTN --speculative-num-steps "$SPEC_STEPS" \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens "$((SPEC_STEPS + 1))" \
    --speculative-token-map /token-map/hot_vocab.pt \
    --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 \
    --cuda-graph-backend-prefill disabled --disable-flashinfer-autotune \
    --ple-offload-embedding --ple-offload-backend file --ple-offload-dir /ple/cache/c1-prod \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder >/dev/null

log "loading (about 11-12 min)"
deadline=$((SECONDS + 7200))
until curl -fsS -o /dev/null "$URL/health" 2>/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
    echo "container exited; see: docker logs $CONTAINER" >&2
    exit 1
  fi
  if [ $SECONDS -gt $deadline ]; then
    echo "not healthy after 120 min; see: docker logs $CONTAINER" >&2
    exit 1
  fi
  sleep 10
done
docker logs "$CONTAINER" 2>&1 | grep -m1 "Deterministic dense routing" || true
log "ready at $URL (OpenAI-compatible API under $URL/v1)"
