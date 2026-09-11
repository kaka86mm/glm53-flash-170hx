# GLM-5.3-Flash Serving on 4× CMP 170HX

Production deployment of GLM-5.3-Flash (320B-parameter MoE, 18B active, natively
multimodal) on four NVIDIA CMP 170HX GPUs (SM80, PCIe Gen2 x4, no P2P/NVLink):

- Pipeline parallelism 4 (PP4); TP is not viable on this interconnect
- EXL3 4bpw weights with offline-generated Marlin INT4 decode sidecar
- DFlash2 speculative decoding, k=7
- 512K context window, 1.53M-token KV cache pool
- Docker-based build and deployment

This repository is a deployment overlay for
[hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx-glm53-flash-exl3-optimized)
(referred to below as *upstream*). Upstream does not deploy correctly on this
hardware as-is; the patches in `patches/` fix the failure modes encountered and
are described in [Technical modifications](#technical-modifications). The
`exllamav3` dependency is upgraded from 0.0.43 to 1.4.8, which improves prefill
throughput by 5–8×.

## Results (4× CMP 170HX 64 GB, 200 W power cap)

| Metric | Value |
|---|---|
| Prefill throughput | 2.0k–3.9k tok/s (16k ctx: 2,040; 64k: 2,627; 131k: 3,926) |
| Single-stream decode | json 97.8 / code 73.9 / math 41–51 / prose 23.7 tok/s |
| Aggregate decode | N1=54 / N4=153 / N8=185–213 / N16=188 tok/s |
| Speculative acceptance length | up to 8.0/8 (DFlash2 k=7; saturated on json/counting) |
| Streaming latency | TTFT 0.4 s (short prompt); ITL mean 66 ms, p95 70 ms |
| Long context | needle retrieval at 64k/128k/256k (95% depth) all pass; 256k cold prefill 90 s |
| Max output | 64K tokens (24K continuous generation verified at 31 tok/s) |
| KV cache pool | 1,533,693 tokens (2.93 concurrent 512K requests) |
| Production acceptance | quality 6/6, tool calling, reasoning separation, vision, 8-way 6-minute soak with 0 errors |

Reference points on identical hardware: an NVFP4 stack (prefill 3.7k tok/s,
single-stream 36–44 tok/s, unstable beyond 64K context) and an AWQ stack without
speculative decoding (~40 tok/s). This stack matches their prefill while
doubling decode/concurrency throughput and extending the context window to 512K.

## Requirements

- 4× GPUs with compute capability 8.0, ≥64 GB each (CMP 170HX unlocked to
  64 GB; driver 610.43.03 open + cmpunlocker)
- PCIe Gen2 x4 links are sufficient: PP4 performs no per-layer all-reduce.
  TP4 on this topology reaches only ~800 tok/s prefill and is not supported.
- ≥229 GB system RAM (page-cache peak during loading)
- ~530 GB disk: EXL3 checkpoint 164 GB + Marlin sidecar 151 GB + DFlash2 draft
  2.3 GB + container image
- Upstream source tree + the patches in this repository (or the Dockerfile
  included here, which already contains the build fixes)

## Repository layout

```
patches/    Three diffs against upstream HEAD (apply in numbered order)
deploy/     build.sh (image build), launch_exl3.sh (serving),
            lanes2.sh (sidecar generation across 4 GPUs)
bench/      Spec-decode, concurrency, needle, prefill-ladder, and
            production-acceptance test scripts
docs/       results.md — full benchmark tables
```

## Quick start

```bash
# 1) Obtain upstream source and apply patches
git clone https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized.git
cd vllm_170hx_glm53-flash-exl3-optimized
git apply ../patches/001-*.patch ../patches/002-*.patch ../patches/003-*.patch

# 2) Build the image (first build ~5 h on 28 cores)
#    The Dockerfile installs the flashinfer 0.6.17 wheel from a local copy
#    (place it under fi/ beforehand; see 003 patch for rationale).
bash ../deploy/build.sh

# 3) Download weights
#    EXL3 checkpoint (164 GB): Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw on ModelScope
#      (byte-identical config to brandonmusic/GLM-5.3-Flash-tr3-4bpw on Hugging Face)
#    DFlash2 draft (2.3 GB): incoai/GLM-5.3-Flash-DFlash2
#      (CC BY-NC-ND — review the license before commercial use)

# 4) Generate the Marlin sidecar (42 layers × 3.6 GB, ~25 min on 4 GPUs)
bash ../deploy/lanes2.sh

# 5) Launch (512K window, util 0.94, DFlash2 k=7, 200 W power cap)
bash ../deploy/launch_exl3.sh
```

## Technical modifications

All modifications are diffs against upstream HEAD and are contained in
`patches/`. Each one addresses a concrete, reproduced failure:

1. **EXL3 tensor release after sidecar swap (patch 001, `exl3.py`)**
   Upstream replaces the per-layer EXL3 parameters with empty tensors after
   the Marlin sidecar is live, but the original tensors are not released
   promptly, adding a net ~3.6 GB per MoE layer. Loading fails with a
   deterministic OOM at the 8th MoE layer (reproduced 3/3). The patch forces
   `gc.collect()` + `cuda.synchronize()` + `empty_cache()` after the swap;
   per-device memory then stays flat at ~41–44 GB through all 42 layers.

2. **`VLLM_PRETEND_NO_DEEP_GEMM=1` (patch 002, `import_utils.py`)**
   The container image builds the vendored DeepGEMM with SM80 in its arch
   list, making `has_deep_gemm()` return True; its attention APIs assert
   SM90+ at runtime (observed as a startup abort in
   `deepgemm-src/csrc/apis/attention.hpp:270`). The patch adds an
   environment hook to `has_deep_gemm()` that reports the package as absent,
   routing all call sites to their Triton fallbacks — the same configuration
   as the upstream author's environment, where DeepGEMM is not installed.

3. **Real `exllamav3.model.config` loading and search-path ordering
   (patch 001, `_exl3_module()`)**
   Upstream installs namespace stubs into `sys.modules` to bypass the
   `exllamav3` package initializer, including a synthetic
   `exllamav3.model.config` that only defines a dummy `Config`. Under
   exllamav3 ≥1.4, `LinearEXL3.__init__` lazily imports `NullConfig` from
   that module, which fails against the stub. Additionally, importing the
   real `config.py` pulls in `ext.py`, which invokes a JIT rebuild of the
   extension unless the prebuilt `.so` directory is already on `sys.path`
   (the runtime image has no nvcc, so a JIT build cannot succeed). The patch
   (a) inserts the prebuilt-extension directory on `sys.path` *before* the
   stub installation and (b) loads the real `config.py` from file via
   importlib, activating `ext.py`'s precompiled-extension branch.

4. **Per-rank H2D staggering (`VLLM_EXL3_H2D_STAGGER_S=12`, patch 001)**
   Concurrent sidecar host-to-device transfers across the 4 pipeline ranks
   trigger PCIe link retrains on Gen2 x4 that surface as asynchronous
   out-of-bounds writes (Xid 31, observed with each rank failing at a
   different layer). The patch delays each rank's first sidecar transfer by
   `rank × stagger` seconds. This is a mitigation specific to this
   interconnect; on NVLink systems it is unnecessary but harmless.

5. **Dockerfile build fixes (patch 003)**
   - Upstream pins FlashInfer `v0.6.18rc10`, whose release lacks the cubin
     wheel (HTTP 404 during build); pinned to 0.6.17 and installed from a
     locally provided wheel to avoid transient network failures.
   - `max_jobs` defaults to 2 in the upstream Dockerfile; raised to 20
     (≈10× faster CUDA compilation on multi-core hosts).
   - The 500 MB wheel-size check is disabled (`RUN_WHEEL_CHECK=false`),
     relevant only for PyPI publication.
   - AlmaLinux/PyPI mirrors are configured for builds in mainland China.

6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (runtime
   environment, from the upstream author's recipe)

### exllamav3 0.0.43 → 1.4.8

Upstream pins exllamav3 0.0.43. Upgrading to
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) 1.4.8
requires modification 3 but is otherwise API-compatible
(`BC_BlockSparseMLP` relocated to `libtorch/blocksparse_mlp_bc.h`). Measured
effect: prefill 480 → 2,040–3,926 tok/s (5–8×) with no change in decode or
concurrency throughput.

### Speculative depth selection (DFlash2)

Measured across k = 2/4/6/7 (tok/s): json 43→70→92→98, code 39→54→55→74,
math peaks at k=6 (51), prose is insensitive (~23, acceptance 1.6). Default
k=7; k=6 is preferable for math-dominant workloads. The KV pool shrinks
slightly with k (k4 = 1.37M / k7 = 1.25M tokens at util 0.92; 1.53M at
util 0.94).

## Benchmarks

Full tables in [docs/results.md](docs/results.md): prefill curve, k sweep,
concurrency, needle retrieval, streaming latency, 64K output, KV pool by
configuration, and the production acceptance scorecard (measured through a
litellm gateway on the serving path).

## Known limitations and operations

- **GPU wedge after a crash.** An Xid 31 fault leaves the affected GPU in a
  state where `cudaSetDevice` fails (`cudaErrorDevicesUnavailable`) until
  power is removed; `nvidia-smi -r` is not supported on this product.
  Recovery is a cold power cycle (~10 min including engine restart).
  Observed frequency ≈1/10 of engine starts.
- **util 0.94 is the ceiling.** At 0.95+ the loading and long-prefill phases
  fail with stochastic out-of-bounds writes (memory-headroom limit of the
  Gen2 x4 topology on this board).
- **Kernel upgrade hazard.** Unattended upgrades install kernels without the
  cmpunlocker modules. `GRUB_DEFAULT=saved` plus `grub-set-default` pin the
  boot to a driver-bearing kernel; on a `driver not loaded` symptom, check
  `uname -r` first.
- **Native MTP is not wired up.** The checkpoint ships a complete NextN
  layer (layer 45, EXL3-quantized), but the vLLM MTP draft path requires
  porting `_make_fused_bsz1` to the 1.4.8 `BC_BlockSparseMLP` constructor
  (signature grew from ~45 to ~55 arguments, adding z/hyper-connection
  bundles); estimated at half a day of work. Reference measurements also
  show the DFlash2 draft outperforming MTP on this model family.
- **Small `max_tokens` budgets.** The model always emits reasoning before
  content; with `max_tokens < 100` the response may contain reasoning only.
  Clients should request ≥400 tokens or rely on gateway-injected defaults.
- **Vision.** Color identification, OCR, counting, and axis-calibrated chart
  estimation all pass. Test fixtures must keep axis labels consistent with
  drawn geometry — the model reads the chart as drawn, not as intended.

## Production integration

The engine exposes an OpenAI-compatible API on port 8093 (served model name
`glm-flash`). In the reference deployment it is fronted by a litellm gateway
under a pre-existing model alias, requiring no client changes. Verified
end-to-end through the gateway: tool calling, reasoning separated into a
`reasoning` field (normalizable to `reasoning_content` at the gateway), and
`chat_template_kwargs` (`enable_thinking`, `reasoning_effort`,
`clear_thinking`).

## Acknowledgments

- [hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx-glm53-flash-exl3-optimized) — base fork
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3 format and kernels; the 1.4.8 MGEMM scheduling work is the source of the prefill improvement
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — SM80 backport ecosystem and MTP fix references
- [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) — the 8-way NVFP4 deployment that informed this work
- dkpoulsen's 170HX lab notes — loading-serialization and crash-recovery practices

## License

Apache-2.0 (derivative of vLLM). Model weights are governed by their own
licenses: the EXL3 pack under ShapleyMCG, the DFlash2 draft under
CC BY-NC-ND — review both before commercial use.
