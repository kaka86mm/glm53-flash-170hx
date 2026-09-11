# GLM-5.3-Flash on 4x CMP 170HX — EXL3 / PP4 / Marlin sidecar / DFlash2

在 4 张 NVIDIA CMP 170HX（SM80 矿卡，PCIe Gen2 x4，无 P2P/NVLink）上以生产质量服务
GLM-5.3-Flash（320B MoE，18B 激活，原生多模态）：流水线并行 4、EXL3 4bpw 权重 +
Marlin INT4 decode sidecar、DFlash2 k=7 投机解码、**512K 上下文 / 153 万 token KV 池**。

本仓库 = 部署配方 + 关键补丁 + 基准与生产验收全记录，基于
[hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized)
（下称 upstream）改造。upstream 在本机四卡上无法直接部署（加载期必 OOM / DeepGEMM
断言 / 内核路径错误），本仓库的六个补丁使其达到生产可用。

## 性能（4×CMP 170HX 64GB，200W 功耗墙）

| 指标 | 数值 |
|---|---|
| Prefill | 2.0k–3.9k tok/s（16k=2,040 / 64k=2,627 / 131k=3,926）|
| 单流解码 | json 97.8 / code 73.9 / math 41–51 / prose 23.7 tok/s |
| 并发聚合 | N1=54 / N4=153 / N8=185–213 / N16=188 tok/s |
| 投机验收长度 | 最高 8.0/8（DFlash2 k=7；json/counting 饱和）|
| 流式延迟 | TTFT 0.4s（短提示）；ITL mean 66ms / p95 70ms |
| 长上下文 | needle 64k/128k/256k（95% 深度）全命中；256k 冷启 90s |
| 最大输出 | 64K（24K 连续生成实测，31 tok/s 持速）|
| KV 池 | 1,533,693 token（512K 满窗 ×2.93 路）|
| 生产验收 | 质量 6/6 + 工具调用 + thinking 分离 + 视觉 + 8 路×6min 浸泡 0 错误 |

对比参考：同为 4×170HX 的 NVFP4 栈（prefill 3.7k、单流 36–44、64K 窗口不稳）与
AWQ 无投机栈（~40 tok/s），本栈在保持 prefill 同档的同时把解码/并发翻倍，且窗口达 512K。

## 硬件 / 软件要求

- 4× compute capability 8.0 GPU，每卡 ≥64GB（CMP 170HX 解锁 64GB；驱动 610.43.03 open + cmpunlocker）
- PCIe Gen2 x4 即可（PP4 无逐层 allreduce；TP 在此拓扑上 prefill 仅 ~800 tok/s，不可用）
- ≥229GB 系统内存（加载期 page cache 峰值）；磁盘 ~530GB：EXL3 权重 164G + Marlin sidecar 151G + DFlash2 草稿 2.3G + 镜像
- upstream 源码 + 本仓库 patches（或直接用本仓库 Dockerfile）

## 快速开始

```bash
# 0) 取 upstream 源码并打补丁（或直接用含 patched Dockerfile 的本仓库树）
git clone https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized.git
cd vllm_170hx_glm53-flash-exl3-optimized
git apply ../patches/*.patch          # 三个补丁按序

# 1) 构建镜像（首次 ~5h；flashinfer wheel 建议先手工下载放进 fi/ 目录，
#    Dockerfile 已改为 COPY 本地 wheel，规避构建期网络抖动）
bash ../deploy/build.sh

# 2) 下载权重（国内走 ModelScope 更快）
#    EXL3 主模型 164G:  Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw（= brandonmusic HF 版，config 逐字节一致）
#    DFlash2 草稿 2.3G: incoai/GLM-5.3-Flash-DFlash2（CC BY-NC-ND，商用前自查许可）

# 3) 生成 Marlin sidecar（42 层 ×3.6G，4 卡并行约 25 分钟）
bash ../deploy/lanes2.sh

# 4) 启动（512K 窗 / util 0.94 / DFlash2 k=7 / 200W 功耗墙）
bash ../deploy/launch_exl3.sh
```

## 相对 upstream 的六个关键补丁

1. **EXL3 张量硬释放（加载期必 OOM 的根因）**
   upstream 在 sidecar 加载后用空 Parameter 替换旧 EXL3 张量，但引用未即时释放，
   净增 3.6GB/层，第 8 个 MoE 层必 OOM。补丁：替换后 `gc.collect + synchronize +
   empty_cache` 强制归还，加载内存曲线平稳在 ~41–44GB/卡。
2. **`VLLM_PRETEND_NO_DEEP_GEMM=1`**
   镜像把 DeepGEMM 编进了 SM80（arch list 含 8.0），其 attention API 运行期断言
   SM90+。给 `has_deep_gemm()` 加 env 钩子模拟"未安装"，所有调用点走 Triton 回退
   ——即 upstream 作者 venv 的真实环境（他没装 DeepGEMM）。
3. **真实 `exllamav3.model.config` 加载 + sys.path 顺序**
   upstream 的 `_exl3_module()` 往 sys.modules 塞假 config stub（只有 dummy
   Config）。exllamav3 ≥1.4 的 `LinearEXL3` 惰性 import `NullConfig` → 假 stub 必炸；
   而真 config.py 又会牵出 ext.py 的 JIT 重编（运行镜像无 nvcc）。补丁：sys.path
   插入预编译 .so 目录 **提前** + importlib 从文件加载真 config.py，走 ext.py 的
   precompiled 分支。
4. **rank 错峰加载（`VLLM_EXL3_H2D_STAGGER_S=12`）**
   Gen2 x4 上 4 rank 并发 sidecar H2D 突发触发链路重训 → 异步野写（Xid 31）。
   每 rank 首次 H2D 前按 rank 序 sleep 错峰。
5. **Dockerfile 构建修复**：flashinfer 钉版 rc10 缺 cubin wheel（404）→ 0.6.17
   本地 wheel COPY；`max_jobs` 默认 2（28 核机器编译慢 10 倍）→ 20；
   wheel 500MB 体积检查对私有镜像无意义 → RUN_WHEEL_CHECK=false；国内 dnf/PyPI 源。
6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（运行 env，作者原配方）。

### exllamav3 版本：0.0.43 → 1.4.8（prefill 5–8×）

upstream 钉的 0.0.43 是化石版。升到真上游 [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
1.4.8 后 API 全兼容（`BC_BlockSparseMLP` 移到 libtorch/blocksparse_mlp_bc.h），
**prefill 从 ~480 提升到 2.0k–3.9k tok/s，解码/并发零变化**——这是本栈最划算的单项升级。

### 投机深度 k 的选择（DFlash2）

实测 k=2→4→6→7：json 43→70→92→**98**，code 39→54→55→**74**，math 在 k=6 峰值
（51），prose 恒 ~23（验收 1.6，无收益无损失）。**默认 k=7**；纯数学负载可退 k=6。
k 越大 KV 池微缩（k4=137 万 / k7=125 万 @0.92；@0.94 为 153 万）。

## 基准全表

见 [docs/results.md](docs/results.md)：prefill 曲线、k sweep、并发、needle、
流式、64K 长输出、生产验收记分卡（含 litellm 全链路）。

## 已知问题与运维手册

- **崩溃会楔卡**：Xid 31 之后该卡 `cudaErrorDevicesUnavailable`，`nvidia-smi -r`
  不支持，唯一解 = 冷断电（rtcwake/手按）。概率约 1/10 boot；楔了就冷启重拉，10 分钟恢复。
- **util 硬墙 0.94**：0.95+ 加载/长 prefill 期随机野写（Gen2 x4 显存余量物理边界）。
- **内核升级陷阱**：unattended-upgrades 装的新内核没有 cmpunlocker 模块。已用
  `GRUB_DEFAULT=saved` + `grub-set-default` 钉回有驱动的内核；再遇
  driver-not-loaded 先 `uname -r`。
- **原生 MTP 未竟**：GLM 自带 NextN 层（layer 45）在 EXL3 checkpoint 中完整，但
  vLLM MTP 路径卡在 1.4.8 的 `BC_BlockSparseMLP` 构造器大改（~45→~55 参，新增
  z/hyper-connection 束），需正式移植半天；且作者数据 DFlash2 线本就快于 MTP 线。
- **小 max_tokens 陷阱**：模型总是先思考。`max_tokens<100` 时 content 可能为空、
  答案落在 reasoning 里。客户端保持 ≥400 或由网关注入默认值。
- **多模态**：颜色/OCR/计数/图表刻度估值全过（图表测试需坐标轴与柱高一致，否则
  模型会"正确地"读出你画错的值——别问我怎么知道的）。

## 生产接入

后端暴露 OpenAI 兼容 API（:8093，model name `glm-flash`）。我们经 litellm 网关以
既有模型名（`dsv4-vision`）对外，存量客户端零改动；工具调用 / thinking 分离
（字段 `reasoning`，网关可归一为 `reasoning_content`）/ `chat_template_kwargs`
（`enable_thinking`、`reasoning_effort`、`clear_thinking`）均经全链路验收。

## 致谢

- [hyd998877/vllm_170hx_glm53-flash-exl3-optimized](https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized) — 本仓库的基座
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3 格式与内核（1.4.8 的 MGEMM 改进是 prefill 质变的来源）
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — sm80 backport 生态与 MTP 修复参考
- [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) — 8 卡 NVFP4 路线的启发
- dkpoulsen 的 170HX 实验记录（flock/加载串行/楔卡恢复经验）

## English summary

GLM-5.3-Flash (320B MoE, natively multimodal) served production-grade on 4× CMP 170HX
mining cards (SM80, PCIe Gen2 x4, no P2P): PP4 + EXL3 4bpw + Marlin INT4 decode
sidecar + DFlash2 k=7 speculative decoding, 512K context, 1.53M-token KV pool.
This repo is a deployment overlay on hyd998877's fork with six essential patches
(EXL3 tensor hard-release fixing a deterministic load-time OOM, DeepGEMM SM80
assertion bypass, real-config loading for exllamav3≥1.4, per-rank H2D staggering
for Gen2 link stability, Dockerfile build fixes) plus an exllamav3 0.0.43→1.4.8
upgrade worth 5–8× prefill (480→3,926 tok/s @131k). Includes full benchmarks,
k-sweep data, a production acceptance suite, and an ops runbook (wedge recovery,
kernel pinning, util walls). Single-stream 23–98 tok/s by workload, 8-way
aggregate 185–213 tok/s, TTFT 0.4s, ITL p95 70ms, zero errors in a 6-minute
8-way soak.

License: Apache-2.0 (derivative of vLLM). Model weights under their own licenses
(EXL3 pack: ShapleyMCG; DFlash2 draft: CC BY-NC-ND — check before commercial use).
