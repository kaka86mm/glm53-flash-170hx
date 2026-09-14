#!/usr/bin/env bash
# PRODUCTION (09-14 v24 A-pack winner): +48-80% C1, +20-37% N8, warm start 9.3min; rollback = launch_exl3_v22_rollback.sh
#   A1  FULL_AND_PIECEWISE CUDA graphs (NO NCCL pinning: tested 09-14, graphs stable
#        unpinned on PP4; pinning PROTO=Simple cost -7~15% cold prefill (large PP sends)
#        zebgop same-hardware PP4: decode 45->66-70 C1, 110->158 C4)
#   A2  DFlash2 greedy drafting (temp-0 traffic +13-21%, tonyd2wild)
#   A3  host performance governor + GPU clock lock (~-10% step time)
#   B1  strided KDA recurrent inputs (vLLM PR #55736 port, in image v24)
#   +   persistent triton/vLLM compile caches (startup acceleration)
# Rollback: bash ~/tools/launch_exl3.sh  (v22 semantics, unchanged)
set -euo pipefail
IMAGE=exl3-glm53:v24
MODEL=/data/models/glm53-exl3-tr3-4bpw
DFLASH=/data/models/glm53-flash-dflash2
MARLIN=/data/models/glm53-exl3-marlin
TEMPLATE=/home/matri/exl3/chat_templates/glm53-enable-thinking-switch.jinja
PORT=8093
OFFLOAD_BYTES=137438953472

# --- guard: marlin sidecar must exist (deleted twice by disk cleanups) ---
SIDECAR_N=$(ls $MARLIN/layer-*.safetensors 2>/dev/null | wc -l)
if [ "$SIDECAR_N" -ne 42 ]; then
  echo "ABORT: marlin sidecar incomplete ($SIDECAR_N/42). Regenerate first:"
  echo "  docker run ... convert_exl3_to_marlin.py --layer 3 (and 42); then ~/lanes2.sh"
  exit 1
fi

# --- A3: host governor + GPU clocks (revert: cpupower frequency-set -g schedutil; nvidia-smi -rgc -rlmc) ---
sudo cpupower frequency-set -g performance >/dev/null 2>&1 || true
for i in 0 1 2 3; do
  sudo nvidia-smi -i $i -pl 200 >/dev/null 2>&1 || true
  # Floor at the 180W-sustainable clock (club-170hx), ceiling at 1455; locking
  # the floor higher would fight the 200W cap. -lmc pins HBM at stock max to
  # prevent the decode-time memory-clock droop.
#  REMOVED: -lgc lock measured NEGATIVE on this box (79 vs 158 tok/s counting); unlocked DVFS wins. sudo nvidia-smi -i $i -lgc 1140,1455 >/dev/null 2>&1 || true
#  sudo nvidia-smi -i $i -lmc 1728 >/dev/null 2>&1 || true
done

sudo rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
docker rm -f glm-exl3 >/dev/null 2>&1 || true
for _i in $(seq 1 36); do
  docker ps -a --format "{{.Names}}" | grep -qx glm-exl3 || break
  sleep 5
done

# --- startup acceleration: persistent kernel/compile caches across containers ---
mkdir -p /data/vllm_caches/triton /data/vllm_caches/vllm

# --- A1: NCCL pinned for full-graph replay on Ampere ---
docker run -d --name glm-exl3 --runtime=nvidia --gpus all --ipc=host \
  -p $PORT:$PORT \
  -v $MODEL:/model:ro -v $DFLASH:/dflash2:ro -v $MARLIN:/marlin \
  -v /path/to/repo/chat_templates:/chat_templates:ro \
  -v /data/vllm_caches/triton:/root/.triton \
  -v /data/vllm_caches/vllm:/root/.cache/vllm \
  -e VLLM_EXL3_MARLIN_DIR=/marlin \
  -e VLLM_EXL3_MARLIN_LAYERS=3-44 \
  -e VLLM_EXL3_NON_ROUTED_FP8=1 \
  -e VLLM_EXL3_NON_ROUTED_FP8_SCOPE=official \
  -e VLLM_EXL3_H2D_STAGGER_S=12 -e VLLM_PRETEND_NO_DEEP_GEMM=1 -e VLLM_EXL3_OUTER_GRAPH=1 \
  -e VLLM_PP_LAYER_PARTITION=13,11,11,10 \
  -e VLLM_PP_DECODE_PHASE_POLICY=pairpack \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_OFFLOAD_MMAP_MIN_BYTES=$OFFLOAD_BYTES \
  -e GLM53_ADAPTIVE_K=ema -e GLM53_ADAPTIVE_K_SET=4,7 \
  -e GLM53_ADAPTIVE_K_ALPHA=0.15 -e GLM53_ADAPTIVE_K_MARGIN=1.0 \
  $IMAGE \
  /model --served-model-name glm-flash --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 1 --pipeline-parallel-size 4 \
  --max-model-len 524288 --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 256 --block-size 256 \
  --gpu-memory-utilization 0.93 --kv-cache-dtype auto --trust-remote-code \
  --mm-processor-cache-gb 0 \
  --chat-template /chat_templates/glm53-enable-thinking-switch.jinja \
  --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-prefix-caching --prefix-caching-hash-algo xxhash --prefix-match-unit 4 --linear-backend marlin --async-scheduling --jit-monitor-mode warn \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40,45,48,50,55,56,60,64],"max_cudagraph_capture_size":64,"cache_dir":"/root/.cache/vllm/torch_compile_cache"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"blocks_per_chunk":4,"cpu_bytes_to_use":'$OFFLOAD_BYTES',"store_threshold":2}}' \
  --speculative-config \
    '{"method":"dflash","model":"/dflash2","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","rejection_sample_method":"standard","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'
echo "glm-exl3 (v24 A-pack) launched on :$PORT"
echo "A/B notes: greedy<->probabilistic via the speculative-config line;"
echo "  graph fallback if capture OOM: revert mode to FULL_DECODE_ONLY + old sizes [3,6,9,12,15,18], or drop util to 0.92"
