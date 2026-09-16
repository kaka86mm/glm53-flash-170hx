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

## store_threshold=2（v22+，冷 prefill A/B）

`kv_connector_extra_config.store_threshold=2`：块被提供两次才写入 CPU 层（一次性上下文不再付存储税，多轮 agent 前缀第二轮入层）。冷 prefill 空池 A/B（122k 加盐 ×2-4 次）：threshold=0 ≈ 2,079 tok/s vs threshold=2 ≈ 1,990 tok/s 中位——打平（噪声内，低样本与后台下载的磁盘竞争相关）。**保留开启**：满池稳态下存储/淘汰税才是主要成本（池压下同负载曾观测 1,256 tok/s），且减少一次性上下文的 CPU 层churn。验证：`kv_offload_stores_skipped` 计数器活跃。

---

## v24（生产现行，2026-09-14 定档）：FULL 图 + greedy drafting + strided KDA + 编译缓存

配置变更 vs v22：`cudagraph_mode=FULL_AND_PIECEWISE`（capture 列表扩至 64、util 0.94→0.93）、DFlash2 `draft_sample_method=greedy`、补丁 007（strided KDA 递归输入）、Triton/vLLM 编译缓存持久化挂载（`vllm_caches/{triton,vllm}`）、host `performance` governor。

### 全口径对比（同机同负载实测，v22 档案 vs v24）

| 指标 | v22 | v24 | 提升 |
|---|---|---|---|
| C1 counting / json | 94.6-102.6 / 96.8-100.7 | **158-163 / 158-161** | +58% / +62% |
| C1 code / prose | 62.6-70.6 / 21.8-23.7 | **96-100 / 40-42** | +48% / +78% |
| C1 math | 38.4-43.0 | **67-86** | +75% |
| 单流长文 32k / 128k | 47-48 / 42-45 | **69-70 / 68-70** | +47% / +55% |
| N4 / N8 聚合 | 153 / 185-213 | **199 / 241-256** | +30% / +20-37% |
| 6×48k agent 型聚合 | 262-267 | **265-272** | +1~2% |
| 冷 prefill 122k | 2,078-2,084 | **2,077** | 持平 |
| 热启动 | 15-16 min | **9.3 min** | -40% |
| 质量/稳定 | 全绿 | 全绿（0 错误/0 NaN） | ✓ |

归因备注：单流提升主力是 FULL 图模式（PP4 的逐层 Python 派发瓶颈消除，与 zebgop-ops 同硬件观察一致）；greedy drafting 对 temp-0 流量增益明确。

### 实测否决项（重要负结果，防止重蹈）

- **`-lgc` 锁频是负优化**：锁 1140-1455 时 counting 79 tok/s，解锁后 158-163（同实例干净归因）。本机 DVFS 解锁优于锁频；governor=performance 保留。
- **NCCL `Ring/Simple` 钉死在 PP 拓扑上有害无益**：FULL 图在 PP4 无钉死稳定回放（钉死药方属 TP allreduce 场景）；`PROTO=Simple` 令 16MB 级 PP 隐状态传输变慢，造成冷 prefill -7~15%，移除后 prefill 回 2,077 且 decode 全保持。

## 已知问题修复：CPU KV 卸载在 preempt/abort 路径崩溃（patch 008，2026-09-16）

**症状**：客户端中途取消/网关超时（或并发突发准入触发连接器内部 preemption）时，引擎在
`transfer_async` 的 `assert len(group_sizes) == len(self.layer_refs_per_group)` 上崩溃（7 != 8），
容器退出；放宽该断言后崩溃点上移到调用方 `assert success`。

**根因**：调度器侧的组视图来自全局 `kv_cache_config.kv_cache_groups`，而 worker 侧
`layer_refs_per_group` 来自本 rank 的实际层缓存——PP 异构混合模型上最后一个 rank 的组数
可以多 1（PR #50653 只统一了块数，没统一组结构）。preempt/abort 构造的 store spec 带 7 组，
撞上初始化为 8 组的 worker。

**修复（防御式，两层）**：`transfer_async` 组数不匹配 → 带计数的警告 + `return False`；
connector 侧三处提交循环（handle_preemptions 的 store、start_kv_transfers 的 store 与 load）
收到 False → 警告 + `mark_completed(job_id)` 丢弃（沿用同函数 non-writer 路径的先例）。
`wait()` 不会挂起（`_transfer_events.get()` 跳过未知 id）。被丢弃的仅是被抢占/中止请求的
一个存储块——CPU 层少一块、后续重算，无害。

**给部署者的提示**：该修复让引擎不再死于该路径，但每次触发会有一条
`kv_offload ... dropped: submit refused` 警告日志——它是组视图分歧的现成遥测，
出现频繁时值得把日志发回来做组结构层面的根治。运行时验证（abort 压测复现脚本见
issue 引用）待下一个维护窗口执行；本补丁经代码级推演 + 与报告者分析互证。
