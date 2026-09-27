# Qwen3.8-Flash-Next on one DGX Spark — fast and bitwise reproducible

A recipe for serving [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
on a single NVIDIA DGX Spark (GB10, 128 GB) with SGLang and MTP speculative decoding, tuned for
single-stream decode speed **without giving up determinism**: the same request gives the same
tokens every time, with or without a prefix-cache hit. Because of that, you can check that your
setup reproduces ours exactly with `scripts/verify.sh`.

## Results (single stream, greedy)

| Workload | Decode | ms / verify | Accept length |
|---|---|---|---|
| C1: 4 fixed prompts, 128 tokens | ~49.5 tok/s | ~60.0 | 3.01 |
| Long prompts 4K–33K, 128 tokens (12 lengths) | ~57.9 tok/s | ~67.1 | 3.80 |
| Rough guide by content | code/YAML/math raw text 57–59, English chat ~42, Japanese chat ~40 tok/s | | |

262K-token prompt: first-token time ~116 s cold, **0.33 s** when resent (259,968 tokens served from
the prefix cache, same output).

Decode speed depends mostly on how many draft tokens are accepted, so it varies with the content.

## What is different from a stock setup

- **SGLang fork** [`d3le/sglang@release/qwen38-spark`](https://github.com/d3le/sglang/tree/release/qwen38-spark):
  18 commits on upstream `113f6f080`, including fixes for nondeterminism and correctness bugs
  (QSA `fast_topk` order, MTP sparse-index holes, a QSA pending-ring collision under speculative
  verify, chunk-size dependent GDN/MoE kernels), deterministic hyperconnection GEMMs, decode tile
  tuning, prefix-exact mamba checkpoints, and a hot-vocabulary draft head.
- **Model**: the NVIDIA NVFP4 checkpoint plus two FP8 conversions made locally from the shipped
  calibration: the LM head (C1) and the GDN projections of 36 of 48 layers (C2).
- **Draft head**: the MTP draft reads a 65,536-token slice of the LM head
  (`data/hot_vocab_mixed_freq_65536.pt`), which saves ~6 ms per verify without changing any output.
- **Deterministic dense routing** (`SGLANG_DETERMINISTIC_BF16_LINEAR`, `SGLANG_FP8_POSITIONAL_PREFILL`):
  a token's value does not depend on batching, chunked prefill or partial prefix hits.

### Determinism contract

| | Guarantee |
|---|---|
| Repeat | Same input and same batch composition → bitwise identical output (also across CUDA graph replay) |
| Prefix cache | A prompt served from a (partial) prefix-cache hit gives the same output as a cold prefill |
| Duplicates | Identical requests in one batch give identical outputs |
| Not guaranteed | Independence from *which other requests* share a batch (monitored, not enforced) |

## Requirements

- DGX Spark (GB10, SM121) with Docker and the NVIDIA container toolkit (DGX OS default)
- ~240 GB free disk: base checkpoint 124 GiB, converted shards ~8 GB, PLE table 48 GiB, kernel cache ~1 GB, image 29 GB
- Nothing else heavy running: the server uses ~104 GB of the shared 128 GB

## Quick start

```bash
git clone https://github.com/d3le/qwen38-flash-next-spark-deterministic.git
cd qwen38-flash-next-spark-deterministic
cp .env.sample .env            # optional: change QWEN38_ROOT (default ~/qwen38)
docker pull ghcr.io/d3le/qwen38-flash-next-spark-deterministic:v0.1
scripts/download.sh            # 124 GiB, sha256-verified against the pinned revision
scripts/convert.sh             # ~10 min, sha256-verified
scripts/start.sh               # ~12 min to healthy (the first start also writes the 48 GiB PLE table)
scripts/verify.sh              # ~5 min: compares outputs with data/reference.json (--all-lengths: ~6 min)
scripts/bench.sh               # decode speed
scripts/stop.sh
```

The server speaks the SGLang and OpenAI-compatible APIs on `http://127.0.0.1:30000`
(`/v1/chat/completions`, reasoning parser `qwen3`, tool-call parser `qwen3_coder`).
Set `HOST=0.0.0.0` in `.env` to serve other machines.

### What `verify.sh` tells you

`ALL PASS` means your machine produced exactly the reference token ids for the fixed prompts,
long prompts (4K–33K) and the 262K prompt (cold and warm). We have verified this on one DGX Spark,
including a from-scratch run of these scripts (fresh cache, PLE table and converted model);
whether other units reproduce it bitwise is exactly what we would like to learn — please open an
issue with your `verify.json` either way. If it fails, the server still works; `bench.sh` numbers
remain meaningful.

## Options

| Setting (`.env`) | Default | Notes |
|---|---|---|
| `SPEC_STEPS` | 3 | 4 draft steps: identical outputs; +6.6% on long prompts, +2% on C1, but −3% on chat/Japanese |
| `PORT`, `HOST` | 30000, 127.0.0.1 | |
| `QWEN38_ROOT` | `~/qwen38` | base, converted model, PLE table, caches, results |
| `BASE_DIR` | `$QWEN38_ROOT/nvidia-base` | point to an existing download to skip `download.sh` |

## Limitations

- Text only is verified; the vision tower is loaded but image/video input is untested here.
- Up to 4 concurrent requests (CUDA graphs for batch 1/2/4), context 262,144, BF16 KV cache.
- Tuned and verified for GB10 only.

## Repository layout

```
scripts/   download, convert, start/stop, verify, bench
tools/     FP8 converters, benchmark clients, verify/bench drivers (stdlib only)
data/      calibration, hot-vocab map, reference outputs, sha256 lists
workloads/ 262K-token surrogate prompt (token ids)
```

## License

The scripts and tools in this repository are under the Apache License 2.0 (`LICENSE`).
The model weights are under the [NVIDIA Open Model License](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
of the base checkpoint; this repository does not redistribute them, and the converted shards you
build locally remain subject to that license. SGLang is Apache-2.0.
