#!/bin/bash
# Build the serving image from this repo's patched tree.
# Run from the repo root after applying patches/ to the upstream source
# (or use directly if you cloned with this Dockerfile in place).
#
# Env:
#   HTTPS_PROXY   proxy for GitHub fetches during build (DeepEP, flashinfer wheels)
#                 — set to empty if you have direct GitHub access
#   MAX_JOBS      parallel compile jobs (default 20; upstream Dockerfile pins 2!)

set -e
IMG=${IMG:-glm53-exl3:v10}
MAX_JOBS=${MAX_JOBS:-20}

docker build -f docker/Dockerfile \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinux2_28-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04 \
  --build-arg PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY:-} \
  --build-arg HTTP_PROXY=${HTTPS_PROXY:-} \
  --build-arg FLASHINFER_VERSION=0.6.17 \
  --build-arg FLASHINFER_RELEASE_TAG=v0.6.17 \
  --build-arg max_jobs=$MAX_JOBS \
  --build-arg RUN_WHEEL_CHECK=false \
  --target vllm-openai \
  -t $IMG .
echo "built $IMG"
