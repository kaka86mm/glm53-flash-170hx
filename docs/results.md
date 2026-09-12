# 基准全表（4×CMP 170HX @200W，EXL3 v10 / util 0.94 / DFlash2 k=7）

## Prefill（JIT 热态）

| 上下文 | tok/s | TTFT |
|---|---|---|
| 4k | 1,796 | 2.2s |
| 16k | 2,040 | 7.7s |
| 64k | 2,627 | 24s |
| 131k | 3,926 | 32s |
| 256k（needle 冷启）| ~2,700 | 90s |

exllamav3 0.0.43→1.4.8 对比：16k 384→2,040；64k 484→2,627；131k ~480→3,926（5–8×）。

## 解码（单流，temperature 0）

| 负载 | k=2 | k=4 | k=6 | k=7 |
|---|---|---|---|---|
| counting | 42.5 | 28* | 29* | 32* |
| repetition | 35.1 | 40.6 | 59.6 | 45.7 |
| json | 43.2 | 69.9 | 92.2 | **97.8** |
| code | 38.9 | 53.5 | 54.9 | **73.9** |
| math | 33.8 | 46.0 | **50.6** | 41.6 |
| prose | 23.7 | 23.2 | 23.4 | 23.7 |

*counting 为每轮首请求（JIT 冷），验收长度仍 7.84–7.96/8。
验收长度峰值：json 8.0/8、counting 7.84/8、code 5.56、prose 1.6。

## 并发（40s 稳态窗口，混合负载）

| N | 聚合 tok/s | 单路 tok/s |
|---|---|---|
| 1 | 54 | 54 |
| 4 | 153 | 38 |
| 8 | 185–213 | 23–25 |
| 16 | 188 | 12 |

6 分钟 8 路浸泡：264 请求 0 错误，p95 完整延迟 17s（含 thinking）。

## 流式（经 litellm 网关）

TTFT 0.4s（短提示）；ITL mean 66ms / p50 66ms / p95 70ms。

## 长上下文

needle（深度 50%）经生产网关：64k/128k/256k 全 PASS；256k 深度 95% PASS。
64K 输出：24,048 token 连续生成自然收尾（775s，31 tok/s）。

## KV 池

| 配置 | 池（token）| 512K 满窗并发 |
|---|---|---|
| util 0.90 / k=2 | 1,128,721 | 2.15× |
| util 0.92 / k=4 | 1,375,019 | 2.62× |
| **util 0.94 / k=7（生产）** | **1,533,693** | **2.93×** |

## 生产验收记分卡（Mac→litellm→引擎全链路）

| 项 | 结果 |
|---|---|
| 功能质量 6 项（算术/GSM/逻辑/代码/JSON/数值比较）| 6/6 |
| thinking 分离 | reasoning 独立字段，content 零污染 |
| 工具调用 | finish=tool_calls，参数 JSON 正确 |
| 视觉（颜色/图表刻度估值）| 2/2 |
| 长输出 12K | 7,368 token 自然收尾 |
| 鲁棒性（坏模型名/超限 max_tokens/别名）| 干净 400 拒绝 / 别名 ✓ |
| 浸泡 8×6min | 0 错误 |

## util 稳定性谱系（实测）

0.95/0.96 加载与长 prefill 期随机越界写入；**0.94 为生产上限**（v10+硬释放后可达）；
0.90–0.92 保守档。

---

## v16：CPU KV 卸载层（OffloadingConnector + 128 GiB /dev/shm）

配置变更：`--kv-transfer-config OffloadingConnector kv_both (blocks_per_chunk 4, cpu_bytes_to_use 128GiB)` + `--enable-prefix-caching --prefix-caching-hash-algo xxhash --prefix-match-unit 4` + `--max-num-seqs 8 --max-num-batched-tokens 2048`；`expandable_segments` 移除（与 mmap 卸载不兼容）。

| 指标 | v10（GPU-only） | v16（+CPU 层） |
|---|---|---|
| KV 池 | 153.4 万 token | **≈450 万**（GPU 158 万 + CPU ~300 万） |
| 并发 467K 驻留 | 3 路 | **4/4 检索全 PASS** |
| 长文 prefill（131k） | 3,926 tok/s | 2,073–2,369（细哈希+卸载簿记成本） |
| warm 前缀命中 prefill | — | 64k 达 14,670 tok/s |
| 解码/并发 | 基准 | 持平（json 97.8 / code 73.9 / N8 ≈200） |

结论：用 30–50% 冷长文 prefill 换 3 倍 KV 容量与 warm 前缀加速；agent 多轮会话（共享前缀）形态净收益为正。prefix-match-unit 取 4（各注意力组 block 尺寸的 GCD，576 全宽对齐需上游 group-exclusion 特性，见 v17–v21 归档）。

## v22：/dev/shm 崩溃安全 + adaptive-k

**补丁 005（上游 PR #52596 移植）**：SharedOffloadRegion 增加 barrier（gloo 世界组），全部 PP rank mmap 完成后 creator unlink 文件名。SIGKILL/崩溃零残留，消除"陈旧 137 GB mmap 卡死下次启动"故障类；启动日志出现 `Unlinked mmap file` 为生效标志。

**补丁 006（MiaAI-Lab 移植，AGPL-3.0）**：adaptive-k DFlash2。调度器按每请求 EMA（observe→batch_k，异步调度路径）在候选集内选每步验证前缀；cudagraph 按候选 k+1 补捕 uniform decode 图。基准为流式真解码速率（`bench/agent_ab.py` 单流 / `bench/agent_conc.py` 并发；首末 delta 时间戳法，±1% 复测一致性；旧的墙钟扣减法在同场景方差 ±75% 已弃用）：

| 场景（48k 上下文 agent 型推理任务，加盐冷前缀） | k=7 固定 | adaptive {4,7} α0.15 m1.0 |
|---|---|---|
| 6 路并发聚合 | 232–238 tok/s | **262–267（+12~13%）** |
| 单流 32k | 44.9 | 48.4（+8%） |
| 单流 128k | 46.8 | 45.1（−4%） |
| 短文单流（json/counting 等） | 基准 | −1~4% |

机理：单流时解码步为纯延迟瓶颈（步率 13.1–13.3 步/s 与 k 无关，verify token 边际成本≈0，砍 k 只丢接受率）；≥6 并发时 verify 进入算力瓶颈，砍掉 prose 死槽位（slot 5–7 逐位接受率 0.06–0.12）换吞吐净赚。生产档 `{4,7} α0.15 margin=1.0`；调参经 {2,4,7}/{4,7}×α0.15/0.25×m0.5/1.0/2.0 全网格选优。运行时热切换：容器内 `/root/.cache/vllm/glm53_adaptive_k.json`（每 50 步检查 mtime；**缺省字段继承上次值**，`set` 只能取启动集子集；改后需 ≥50 步再核对 `reloaded` 日志）。

功能回归（v22 生产实测）：tool-call（glm47 parser）✓、thinking 分离（`reasoning` 字段独立，content 无泄漏）✓、needle 128k ✓。
