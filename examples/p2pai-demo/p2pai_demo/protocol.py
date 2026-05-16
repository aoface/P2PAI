"""控制 / 数据 / 梯度三平面消息 (架构 §7.4)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class OpCode:
    HELLO = "HELLO"                       # 握手 + 交换 peer 列表
    ANNOUNCE_PROFILE = "ANNOUNCE_PROFILE" # 节点资源画像广播
    RECRUIT = "RECRUIT"                   # 招募加入任务
    JOIN_ACK = "JOIN_ACK"
    REFUSE = "REFUSE"
    SHARD_INVENTORY = "SHARD_INVENTORY"   # 上报本地分片清单
    EXEC_PLAN = "EXEC_PLAN"               # 下发执行计划
    HEARTBEAT = "HEARTBEAT"
    GRAD_PUSH = "GRAD_PUSH"               # D-PSGD 梯度推送 (此处推 LoRA ΔW)
    INFER_REQUEST = "INFER_REQUEST"
    INFER_RESULT = "INFER_RESULT"
    TEARDOWN = "TEARDOWN"


@dataclass
class Envelope:
    """所有线上消息的统一信封, 带签名."""

    op: str
    body: Dict[str, Any]
    sender_peer: str
    sender_pubkey: str         # base64 ed25519 公钥
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    ttl: int = 6               # gossip 转发 TTL
    signature: str = ""        # base64 ed25519 签名 (覆盖 op + body + sender_peer + msg_id + ts)
    task_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "body": self.body,
            "sender_peer": self.sender_peer,
            "sender_pubkey": self.sender_pubkey,
            "msg_id": self.msg_id,
            "ts": self.ts,
            "ttl": self.ttl,
            "signature": self.signature,
            "task_id": self.task_id,
        }

    def signing_payload(self) -> dict:
        d = self.to_dict()
        d.pop("signature")
        d.pop("ttl")  # ttl 在转发中递减, 不参与签名
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        return cls(
            op=d["op"],
            body=d.get("body", {}),
            sender_peer=d["sender_peer"],
            sender_pubkey=d["sender_pubkey"],
            msg_id=d["msg_id"],
            ts=d["ts"],
            ttl=int(d.get("ttl", 0)),
            signature=d.get("signature", ""),
            task_id=d.get("task_id"),
        )
