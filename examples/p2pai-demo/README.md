# P2PAI Demo — 基础 P2P 分布式训推一体最小可运行实现

本目录是对仓库根部 [`README.md`](../../README.md) 中所描述架构的一个**可在单机上跑通**的最小参考实现，
用于演示：

| 架构概念 (README 章节) | 在 demo 中的体现 |
| --- | --- |
| §4.1 用户资源自治层 | `policy.py` + 每个节点独立 TOML 配置 (`configs/*.toml`) |
| §4.2 P2P 网络传输层 | `transport.py` (asyncio TCP) + `gossip.py` (签名+TTL+去重广播) |
| §4.3 单流原生训推一体 | `engine.py` — Streaming LoRA: 同一次 `step()` 既反传更新 LoRA ΔW 又输出生成文本 |
| §4.3.3 D-PSGD 去中心化聚合 | `aggregator.py` — FedAvg-on-LoRA + 陈旧梯度衰减/丢弃 |
| §4.4 任务发起节点临时调度 | `coordinator.py` — 任意节点 `submit_task` 即晋升 Coordinator |
| §4.4.3 三级敏感度分级 (S1/S2/S3) | `shard.py` — 按 layer × module 给每个 LoRA 分片打 sensitivity 标签 |
| §7.1 ShardID / ShardMeta | `shard.py` — 元数据带 CID/version/PVF/hash |
| §7.6 PeerID + 签名 | `identity.py` — Ed25519 + BLAKE2b 派生 PeerID + 消息签名校验 |

> **注意**：demo 用 `asyncio TCP` 简化了真实方案中的 `libp2p`，用 `distilgpt2 (82M)` 代替
> 真实 7B 模型；这些都是为了**在一台普通笔记本的 CPU 上几秒内跑起来**。
> 协议消息格式 (`protocol.py`) 与根 README §7 的 proto 定义一一对应，可直接替换成 libp2p / proto3。

## 1 选用的开源小模型

| 项 | 选择 | 理由 |
| --- | --- | --- |
| 模型 | **`distilgpt2`**（HuggingFace，~82M 参数，~330 MB，MIT 许可，6 层 Transformer Decoder） | CPU 友好，结构与 Llama/Qwen 同族，可直接套 LoRA |
| 微调方法 | **LoRA**（rank=4，注入到 `c_attn` / `c_proj`） | 可训练参数仅 ~203k，便于 P2P 传输（单次序列化 ~1 MB） |
| 优化器 | SGD lr=1e-3 | 简化演示；生产中可换 AdamW |
| 训练目标 | next-token-prediction 自监督（用 prompt 本身做 label） | 真实场景应替换为业务任务损失 |

替换为更强模型只需改 `configs/*.toml` 的 `model_id`（如 `Qwen/Qwen2.5-0.5B`、`HuggingFaceTB/SmolLM2-135M`），
并相应调整 `engine.py` 中 `target_modules`（不同模型族注意力命名不同）。

## 2 目录结构

```
examples/p2pai-demo/
├── README.md                  # 本文件
├── requirements.txt
├── p2pai_demo/                # Python 包
│   ├── identity.py            # Ed25519 PeerID + 签名
│   ├── protocol.py            # 消息信封 + 11 个 OpCode
│   ├── transport.py           # asyncio TCP，长度前缀 JSON 帧
│   ├── gossip.py              # 签名校验 + TTL 转发 + 去重
│   ├── shard.py               # ShardID / ShardMeta / S1-S2-S3
│   ├── policy.py              # 用户策略 (TOML)
│   ├── engine.py              # 单流训推一体 (Streaming LoRA)
│   ├── aggregator.py          # D-PSGD FedAvg-on-LoRA
│   ├── coordinator.py         # 任务级临时调度
│   ├── node.py                # 节点守护进程
│   └── cli.py                 # CLI 入口
├── configs/
│   ├── node1.toml             # 监听 7801 (集群协调器)
│   ├── node2.toml             # 监听 7802，bootstrap=7801
│   └── node3.toml             # 监听 7803，bootstrap=7801,7802
├── data/sample_prompts.txt
└── scripts/
    ├── smoke_test.py          # 离线冒烟测试
    └── run_cluster.sh         # 一键启动三节点集群
```

## 3 快速开始

### 3.1 安装依赖

```bash
cd examples/p2pai-demo
pip install -r requirements.txt
```

第一次会下载 PyTorch CPU wheel (~200 MB) + `distilgpt2` (~330 MB)。

### 3.2 单机冒烟测试（不组网）

```bash
python3 scripts/smoke_test.py
```

预期输出（节选）：

```
[ok] identity sign/verify
[ok] shard inventory: 36 LoRA shards
      sensitivity breakdown: {'S3': 16, 'S1': 20}
      step=0 v=1 loss=7.35 out='...' 136ms
      step=3 v=4 loss=6.60 out='...'  87ms
[ok] streaming train+infer: loss 7.35 -> 6.60
[ok] aggregator weighted average
[ok] aggregator drops stale gradient
[ok] serialize roundtrip (1097188 bytes base64)
*** all smoke tests passed ***
```

### 3.3 三节点 P2P 集群

```bash
bash scripts/run_cluster.sh
# 自定义参数:
RUN_TIME=120 MAX_STEPS=16 LOG_LEVEL=INFO bash scripts/run_cluster.sh
```

预期事件序列（节选自实际日志）：

```
node1 up at 127.0.0.1:7801  peer_id=12D3...
node1 submit task <id> kind=HYBRID
RECRUIT broadcast to 4 peer(s)
[<id>] member joined: 12D3rdBf-9xlWi (group=1)
[<id>] member joined: 12D3u8WeJc-_Br (group=2)
[<id>][step=0] loss=5.0369  out='...'
[<id>] received delta from 12D3... (v=4, pending=1)
aggregated 2 delta(s)
[<id>] imported aggregated weights v=4
[<id>] aggregated + broadcast new version v=4
...
[<id>] worker finished; loss 5.04 -> 8.00
```

观察要点：

1. **训推一体**：每个 `step=N` 既输出 `out=...`（推理结果）又更新 LoRA 权重（version++）。
2. **跨节点权重同步**：`imported aggregated weights v=N` 说明 worker 收到了协调器聚合后的新版本。
3. **动态成组**：`group=1 → group=2` 演示了任务运行中加入新成员。
4. **签名校验**：所有消息均经 Ed25519 验签（错签会被 gossip 层 DROP，可在 `gossip.py:_on_envelope` 注入测试）。

## 4 与架构需求的对应

下表把 demo 落地到根目录 [`REQUIREMENTS.md`](../../REQUIREMENTS.md) 的功能需求：

| 需求 ID | demo 覆盖度 | 文件位置 |
| --- | --- | --- |
| FR-AUT-01/02 | ✅ TOML 策略 + 启动加载 | `policy.py`, `configs/*.toml` |
| FR-AUT-03 | ✅ 策略拒绝训练任务 | `node._handle_recruit` → `REFUSE` |
| FR-NET-01 | ⚠️ gossip 已实现，DHT 用 known_addrs 静态列表占位 | `gossip.py`, `transport.py` |
| FR-NET-03 | ⚠️ 内容寻址 (CID) 已实现，多源并行 fetch 留为 TODO | `shard.py:ShardID.cid` |
| FR-NET-05 | ✅ Ed25519 PeerID + 消息签名；Noise 在 demo 中简化为明文 TCP | `identity.py`, `gossip.py` |
| FR-STO-02 | ✅ ShardMeta 完整字段（含 sensitivity, version, pvf） | `shard.py` |
| FR-RT-01 | ✅ 单流 forward 内反传 + 原地 LoRA 更新 | `engine.StreamingLoRAEngine.step` |
| FR-RT-04 | ✅ FedAvg-on-LoRA + staleness 衰减 + 丢弃 | `aggregator.py` |
| FR-RT-05 | ⚠️ 缺失/错误/动态替换：demo 只实现版本对齐，分级容错路径见 README §4.4.4 |  |
| FR-SCH-01 | ✅ 任务发起节点临时担任 Coordinator | `coordinator.py` |
| FR-SCH-02 | ✅ RECRUIT → JOIN → EXEC_PLAN | `coordinator.submit` |
| FR-SCH-04 | ✅ 心跳超时 EVICT | `coordinator._heartbeat_monitor` |
| FR-SCH-05 | ✅ Coordinator 不中转数据，仅聚合权重 | 见 `_handle_grad_push` |
| FR-SEC-01 | ✅ 所有 Envelope 均签名校验 | `gossip.Gossip.on_transport_message` |

⚠️ = 在 demo 中以简化或占位形式存在，标注了后续打通至生产实现的方向。

## 5 已知限制与后续工作

| 项 | 当前 demo | 生产实现 (与根 README §6 对齐) |
| --- | --- | --- |
| 传输 | asyncio TCP, 明文 JSON | libp2p (rust) + Noise XX + QUIC |
| 节点发现 | 静态 `bootstrap_peers` | Kademlia DHT + mDNS + Bootstrap |
| 分片传输 | LoRA 整体 base64 | CID + BLAKE3 校验 + 多源并行下载 |
| 模型规模 | distilgpt2 (82M) | Llama-3-8B / Qwen2-7B (Q4) |
| 单流算子 | LoRA-only, 基座冻结 | LoRA + 可选基座低频更新, FlashAttn |
| 容错 | 心跳 EVICT | + S1 常驻 + 跨组重组 + Krum/Median 防毒化 |
| 资源隔离 | 仅策略校验 | + cgroups v2 / Windows Job Objects |
| UI | CLI + 日志 | + Tauri 桌面 + Web 拓扑 |

## 6 故障注入与小实验

- **杀 worker**：`pkill -f node2.toml`，观察 coord 5–15 s 后 `EVICT`。
- **网络分区**：用 `iptables -A INPUT -p tcp --dport 7802 -j DROP` 模拟 node2 失联。
- **窜改签名**：在 `gossip.py:_on_envelope` 前临时把 `env.body` 修改一字节，所有消息会被 DROP 并出现 `signature invalid` 警告。
- **策略拒绝**：将 `configs/node3.toml` 中 `allow_training=false`，看 RECRUIT 是否被 `REFUSE`。

## 7 许可证

本 demo 与仓库根部一致采用 MIT 风格许可（详见仓库 LICENSE）。`distilgpt2` 由 HuggingFace 发布，遵循其原始 Apache-2.0 / MIT 许可。
