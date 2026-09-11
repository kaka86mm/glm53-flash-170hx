#!/bin/bash

run_lane() {
  local D=$1; shift
  for L in "$@"; do
    [ -s /data/models/glm53-exl3-marlin/layer-$(printf %02d $L).safetensors ] && continue
    echo "lane$D layer $L start $(date +%H:%M:%S)" >> /tmp/sidecar-gen.log
    docker run --rm --runtime=nvidia --gpus device=$D \
      -v /data/models/glm53-exl3-tr3-4bpw:/model:ro \
      -v /data/models/glm53-exl3-marlin:/marlin \
      -v /repo/scripts:/scripts:ro \
      -e VLLM_EXL3_MARLIN_LAYERS=3-44 \
      -w /vllm-workspace \
      --entrypoint python3 \
      exl3-glm53:v1 /scripts/convert_exl3_to_marlin.py \
      --model /model --out-dir /marlin --layer $L --device 0 > /tmp/conv_$L.log 2>&1
    echo "lane$D layer $L exit=$? $(date +%H:%M:%S)" >> /tmp/sidecar-gen.log
  done
}
run_lane 0 4 8 12 16 20 24 28 32 36 40 44 &
run_lane 1 5 9 13 17 21 25 29 33 37 41 &
run_lane 2 6 10 14 18 22 26 30 34 38 &
run_lane 3 7 11 15 19 23 27 31 35 39 43 &
wait
echo "LANES2_DONE $(ls /data/models/glm53-exl3-marlin/layer-*.safetensors 2>/dev/null | wc -l)" >> /tmp/sidecar-gen.log
