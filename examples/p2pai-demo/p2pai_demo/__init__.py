"""P2PAI demo: 基于 P2P 网络的分布式 AI 推理训练一体化最小可运行实现。

模块映射 (与 README 架构章节对应):
    - identity:    §7.6 Ed25519 PeerID 与消息签名
    - protocol:    §7 控制 / 数据 / 梯度三平面消息格式
    - transport:   §4.2 P2P 网络传输层 (此处用 asyncio TCP 简化 libp2p)
    - gossip:      §4.2 GossipSub 广播 + 去重
    - shard:       §7.1 ShardID / ShardMeta 与 S1/S2/S3 分级
    - policy:      §4.1 用户资源自治层 (TOML 策略)
    - engine:      §4.3 单流原生训推一体执行引擎 (Streaming LoRA)
    - aggregator:  §4.3.3 D-PSGD 去中心化梯度聚合 (FedAvg-on-LoRA)
    - coordinator: §4.4 任务发起节点临时调度
    - node:        节点守护进程 (Rust crate `node` 在 demo 中的 Python 对应)
"""

__version__ = "0.1.0"
