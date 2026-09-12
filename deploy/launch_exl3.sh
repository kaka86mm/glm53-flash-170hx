#!/usr/bin/env bash
# EXL3 tr3-4bpw + DFlash2 k=7 on 4x CMP 170HX (PP4), dockerized.
# v22: + /dev/shm mmap unlink-after-barrier (upstream #52596 port, no more stale-file wedges)
#      + adaptive-k DFlash2 (MiaAI-Lab port): EMA-chosen verify prefix {4,7}, tuned for
#        concurrent long-context agent load (+12% aggregate @6 streams, -4% single 128k).
#        Runtime override: {"mode":"off"} in /root/.cache/vllm/glm53_adaptive_k.json
#        (fields absent from the JSON keep previous values; set must be subset of boot set).
set -euo pipefail
IMAGE=exl3-glm53:v22
MODEL=/data/models/glm53-exl3-tr3-4bpw
DFLASH=/data/models/glm53-flash-dflash2
MARLIN=/data/models/glm53-exl3-marlin
TEMPLATE=/path/to/repo/chat_templates/glm53-enable-thinking-switch.jinja
PORT=8093

for i in 0 1 2 3; do sudo nvidia-smi -i $i -pl 200 >/dev/null 2>&1; done
# v22 unlinks the offload mmap after all ranks map it, so no stale files can remain
# from this image; the rm only guards against leftovers from older (pre-v22) engines.
sudo rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
docker rm -f glm-exl3 >/dev/null 2>&1 || true
# The nvidia runtime takes ~1 min to release a killed engine container; wait
# for the name to actually free up or docker run hits a name conflict.
for _i in $(seq 1 36); do
  docker ps -a --format "{{.Names}}" | grep -qx glm-exl3 || break
  sleep 5
done

docker run -d --name glm-exl3 --runtime=nvidia --gpus all --ipc=host \
  -p $PORT:$PORT \
  -v $MODEL:/model:ro -v $DFLASH:/dflash2:ro -v $MARLIN:/marlin \
  -v /path/to/repo/chat_templates:/chat_templates:ro \
  -e VLLM_EXL3_MARLIN_DIR=/marlin \
  -e VLLM_EXL3_MARLIN_LAYERS=3-44 \
  -e VLLM_EXL3_NON_ROUTED_FP8=1 \
  -e VLLM_EXL3_NON_ROUTED_FP8_SCOPE=official \
  -e VLLM_EXL3_H2D_STAGGER_S=12 -e VLLM_PRETEND_NO_DEEP_GEMM=1 -e VLLM_EXL3_OUTER_GRAPH=1 \
  -e VLLM_PP_LAYER_PARTITION=13,11,11,10 \
  -e VLLM_PP_DECODE_PHASE_POLICY=pairpack \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_OFFLOAD_MMAP_MIN_BYTES=137438953472 \
  -e GLM53_ADAPTIVE_K=ema -e GLM53_ADAPTIVE_K_SET=4,7 \
  -e GLM53_ADAPTIVE_K_ALPHA=0.15 -e GLM53_ADAPTIVE_K_MARGIN=1.0 \
  $IMAGE \
  /model --served-model-name glm-flash --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 1 --pipeline-parallel-size 4 \
  --max-model-len 524288 --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 256 --block-size 256 \
  --gpu-memory-utilization 0.94 --kv-cache-dtype auto --trust-remote-code \
  --mm-processor-cache-gb 0 \
  --chat-template /chat_templates/glm53-enable-thinking-switch.jinja \
  --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-prefix-caching --prefix-caching-hash-algo xxhash --prefix-match-unit 4 --linear-backend marlin --async-scheduling --jit-monitor-mode warn \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12,15,18]}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"blocks_per_chunk":4,"cpu_bytes_to_use":137438953472,"store_threshold":2}}' \
  --speculative-config \
    '{"method":"dflash","model":"/dflash2","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'
echo "glm-exl3 launched on :$PORT — docker logs -f glm-exl3"
