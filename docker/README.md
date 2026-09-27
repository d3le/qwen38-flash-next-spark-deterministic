# Release image

`ghcr.io/d3le/qwen38-flash-next-spark-deterministic:v0.1` is the exact environment the
reference outputs were measured in:

1. The SGLang trial build `sgl-project/sglang@32a6f3bb7` (CUDA 13.0.3, arm64, SM121), with its
   15.7 GiB site-packages layer re-cut into 8 layers of <= 2 GiB so that GHCR accepts it
   (10 GB per-layer limit). A per-file manifest (path, mode, owner, size, sha256) of the image
   before and after the re-cut is identical (245,327 entries).
2. `/sgl-workspace/sglang/python` replaced by `d3le/sglang@release/qwen38-spark` (`194c881d1`);
   sglang is an editable install at that path.
3. `/opt/qwen38/runtime-cache`: the JIT-compiled kernels of the measurement runs (flashinfer
   CUTLASS MoE/GEMM, sgl-kernel JIT, Triton, torch.compile). `scripts/start.sh` copies them into
   the host cache on first start; without them the first start compiles for over an hour.

`Dockerfile` is the last step (2 and 3). Rebuilding step 1 from source gives a different set of
wheels and compiled kernels, so it is not the supported path for bitwise reproduction.
