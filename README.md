# GLM-5.3-Flash Serving on 4× CMP 170HX

Production deployment of GLM-5.3-Flash (320B-parameter MoE, 18B active, natively
multimodal) on four NVIDIA CMP 170HX GPUs (SM80, PCIe Gen2 x4, no P2P/NVLink):

- Pipeline parallelism 4 (PP4); TP is not viable on this interconnect
- EXL3 4bpw weights with offline-generated Marlin INT4 decode sidecar
- DFlash2 speculative decoding, k=7, greedy drafting, with adaptive
  verification length ({4,7} EMA profile)
- FULL_AND_PIECEWISE CUDA graphs covering the whole decode step on Ampere
  (PP4) — the single largest decode lever on this rig
- Strided KDA recurrent inputs (port of vLLM PR #55736): decode KDA consumes
  merged-projection column slices in place, no per-layer packing copies
- 512K context window; KV cache tiered across GPU HBM (~1.58M tokens) and a
  128 GiB pinned /dev/shm CPU tier (~3M tokens) — ~4.5M tokens total
- Crash-safe CPU KV offload region: the backing mmap is unlinked after all
  pipeline ranks map it, so a SIGKILL or wedge cannot leak /dev/shm files
- Persistent Triton/vLLM compile caches across container restarts
  (warm start 15-16 min -> 9.3 min)
- Docker-based build and deployment

This repository is a deployment overlay for
[hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized)
(referred to below as *upstream*). Upstream does not deploy correctly on this
hardware as-is; the patches in `patches/` fix the failure modes encountered and
are described in [Technical modifications](#technical-modifications). The
`exllamav3` dependency is upgraded from 0.0.43 to 1.4.8, which improves prefill
throughput by 5–8×.

## Results (4× CMP 170HX 64 GB, 200 W power cap)

| Metric | v22 (prior) | v24 (current) |
|---|---|---|
| Prefill throughput (cold, 131k) | 2,078–2,084 tok/s | 2,077 tok/s (parity) |
| Prefill throughput (warm prefix hit) | up to 14.7k tok/s @64k | unchanged mechanism |
| Single-stream decode C1 | json 97 / code 63–71 / prose 22–24 tok/s | **json 158–161 / code 96–100 / prose 40–42 / counting 158–163 tok/s** |
| Single-stream long-context decode | 47–48 @32k / 42–45 @128k | **69–70 @32k / 68–70 @128k** |
| Aggregate decode | N4=153 / N8=185–213 | **N4=199 / N8=241–256** |
| 6-way concurrent long-context decode | 262–267 tok/s | **265–272 tok/s** |
| Speculative acceptance length | up to 8.0/8 (DFlash2 k=7) | unchanged |
| Long context | needle 64k/128k/256k pass; 4× 467K resident | unchanged |
| KV cache pool | ~4.5M tokens (1.58M GPU + ~3M CPU tier) | unchanged (util 0.93) |
| Warm restart | 15–16 min | **9.3 min** (persistent compile caches) |
| Production acceptance | quality 6/6, tools, reasoning, vision, 0-error soak | re-verified on v24 (0 errors / 0 NaN) |

Tested and rejected on this rig (documented so nobody retries blind): `-lgc`
clock locking (79 vs 158 tok/s counting, unlocked DVFS wins); NCCL
`Ring/Simple` pinning (unnecessary for FULL graphs under PP; `PROTO=Simple`
slowed 16 MB PP hidden-state transfers, costing 7–15% cold prefill).

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
patches/    Seven diffs against upstream HEAD (apply in numbered order).
            006 is AGPL-3.0 (see License); the rest are Apache-2.0.
deploy/     build.sh (image build), launch_exl3.sh (serving),
            lanes2.sh (sidecar generation across 4 GPUs)
bench/      Spec-decode, concurrency, needle, prefill-ladder,
            production-acceptance, and streaming true-decode-rate
            (agent_ab.py / agent_conc.py) test scripts
docs/       results.md — full benchmark tables
```

## Quick start

```bash
# 1) Obtain upstream source and apply patches
git clone https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized.git
cd vllm_170hx_glm53-flash-exl3-optimized
git apply ../patches/001-*.patch ../patches/002-*.patch ../patches/003-*.patch \
          ../patches/004-*.patch ../patches/005-*.patch ../patches/006-*.patch \
          ../patches/007-*.patch

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

# 5) Size /dev/shm for the CPU KV tier (default tmpfs is too small):
sudo mount -o remount,size=170G /dev/shm

# 6) Launch (512K window, util 0.94, DFlash2 k=7 + adaptive-k, 128 GiB CPU KV
#    tier, 200 W power cap)
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

6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` removed** The upstream
   author's recipe sets this allocator mode, but it is incompatible with the
   mmap-backed CPU KV tier (patch 004): the CUDA allocator's expandable
   segments conflict with pinning/registering views over the shared mmap.
   The launch script deliberately does not set it.

7. **CPU KV offload tier (patch 004, six files)**
   Wires the `OffloadingConnector` into a usable PP4 configuration:
   - a port of vLLM PR #50653 (unify `num_cpu_blocks` across pipeline ranks —
     without it, heterogeneous PP ranks compute different CPU block counts
     and the connector deadlocks at startup);
   - `num_cpu_blocks` plumbed through `KVCacheConfig` and the offloading
     config;
   - the divisibility assertion in `build_offloading_config` relaxed to skip
     ring-buffer groups that never participate in prefix caching
     (e.g. the Kpool tail spec);
   - runtime flags: `--enable-prefix-caching --prefix-caching-hash-algo
     xxhash --prefix-match-unit 4`. The match unit is the GCD of the
     per-group block sizes (576-wide alignment needs an upstream
     "exclude non-participating groups" feature that does not exist yet).
   Effect: KV pool 1.53M → ~4.5M tokens; 4× 467K requests resident;
   cold long-context prefill drops 30–50% (bookkeeping + fine hashing),
   warm shared-prefix prefill rises to 14.7k tok/s.

8. **Crash-safe offload mmap (patch 005, port of vLLM PR #52596)**
   `SharedOffloadRegion` gains a `barrier` (gloo world-group) supplied by
   `CPUOffloadingSpec`; once every pipeline rank has mapped the file, the
   creator unlinks the path. Mappings taken before the unlink stay valid, so
   the kernel reclaims the 128 GiB region when the last worker exits —
   including on SIGKILL or a GPU-wedge hard stop. This eliminates the
   "stale vllm_offload_*.mmap wedges the next start" failure class and, as a
   side effect, name collisions between two engine instances. Merged with
   the local `VLLM_OFFLOAD_MMAP_MIN_BYTES` floor (the creator sizes the file
   to at least the largest rank's expectation, required for heterogeneous
   PP rank layouts).

10. **v24 decode speed pack (patches + configuration)**
    - Patch 007 (port of vLLM PR #55736): the fused KDA recurrent kernel walks
      q/k/v/beta by token stride, so decode consumes merged-projection column
      slices in place instead of four per-layer per-step packing copies; the
      output buffer is allocated dense because the kernel writes o densely.
    - `cudagraph_mode=FULL_AND_PIECEWISE` with a capture list up to 64
      tokens: under FULL_DECODE_ONLY only the smallest decode batches replay
      graphs, larger batches run eager and the PP stage pipeline becomes
      CPU-dispatch-bound. No NCCL pinning: FULL graphs replay stably on PP4
      without it (and PROTO=Simple slows large PP sends).
    - DFlash2 `draft_sample_method=greedy`: at temperature 0 greedy drafts pin
      draft probability to 1, making the probabilistic ratio test strictly
      easier to pass.
    - Persistent compile caches (`/root/.triton`, `/root/.cache/vllm`)
      mounted from the host: warm restarts skip Triton JIT and inductor
      compilation.
    Combined effect on this rig: +48-80% single-stream decode, +20-37% N8
    aggregate, cold prefill parity, warm restart 15-16 -> 9.3 min.

9. **Adaptive verification length for DFlash2 (patch 006, AGPL-3.0, ported
   from MiaAI-Lab)**
   The drafter keeps its trained 8-token block; the scheduler verifies only a
   per-step prefix chosen from a per-request EMA of accepted drafts
   (2/4/7 candidates, batch-uniform, one extra uniform decode CUDA graph per
   candidate length). Env-gated (`GLM53_ADAPTIVE_K`, default off) with a
   runtime JSON override. Production profile on this hardware:
   `{4,7}, alpha=0.15, margin=1.0` — +12–13% aggregate decode at 6-way
   concurrent long-context load, −4% single-stream at 128k. At single-stream
   this rig is latency-bound (step rate is independent of k, so trimming only
   loses acceptance); the win appears where verify compute becomes the
   bottleneck (≥6 concurrent streams). See docs/results.md for the profile
   sweep and the streaming measurement methodology.

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
util 0.94). Patch 006 (adaptive-k) keeps num_speculative_tokens=7 and trims
the verified prefix per step instead — see modification 9.

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

- [hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized) — base fork
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3 format and kernels; the 1.4.8 MGEMM scheduling work is the source of the prefill improvement
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — the adaptive verification length design (patch 006 is a port; that repository is AGPL-3.0)
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — SM80 backport ecosystem and MTP fix references
- [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) — the 8-way NVFP4 deployment that informed this work
- dkpoulsen's 170HX lab notes — loading-serialization and crash-recovery practices

## License

Apache-2.0 (derivative of vLLM), with one exception:
`patches/006-adaptive-k-dflash2.patch` is a derivative of an
[AGPL-3.0-licensed](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
work and remains AGPL-3.0 — review its terms before incorporating it into a
product. Model weights are governed by their own licenses: the EXL3 pack
under ShapleyMCG, the DFlash2 draft under CC BY-NC-ND — review both before
commercial use.
