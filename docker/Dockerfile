# Release image = the SGLang trial build the measurements were made on
# (sgl-project/sglang 32a6f3bb7, CUDA 13.0.3, arm64/SM121, its 16.6 GB
# site-packages layer re-cut into <=2 GiB layers for GHCR, file contents
# unchanged) + the Python sources of d3le/sglang release/qwen38-spark.
# sglang is an editable install pointing at /sgl-workspace/sglang/python, so
# replacing that directory swaps the Python code and nothing else.
#
# Build context: `git archive <release commit> python | tar -x -C ctx` plus
# ctx/runtime-cache (the /root/.cache/sglang of the measurement runs).
ARG BASE=qwen38-nvidia:trial-32a6f3bb7cf5-split
FROM ${BASE}
ARG SGLANG_COMMIT
RUN rm -rf /sgl-workspace/sglang/python
COPY python /sgl-workspace/sglang/python
# Compiled JIT kernels (flashinfer CUTLASS MoE/GEMM, sgl-kernel JIT, Triton,
# torch.compile) from the runs the reference outputs were measured with.
# Without them the first start compiles for over an hour with MAX_JOBS=1.
# scripts/start.sh copies them into the host cache directory on first use.
COPY runtime-cache /opt/qwen38/runtime-cache
LABEL org.opencontainers.image.source="https://github.com/d3le/qwen38-flash-next-spark-deterministic" \
      org.opencontainers.image.description="SGLang for Qwen3.8-Flash-Next NVFP4 on DGX Spark (deterministic, MTP)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.sglang.release.branch="release/qwen38-spark" \
      ai.sglang.release.commit="${SGLANG_COMMIT}"
CMD ["/bin/bash"]
