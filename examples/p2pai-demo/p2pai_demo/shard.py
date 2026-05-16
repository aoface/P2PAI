"""ShardID / ShardMeta 与三级敏感度分级 (架构 §7.1, §4.4.3)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import List


class Sensitivity(IntEnum):
    """参数敏感度三级 (PVF 越高越敏感)."""

    S1 = 0  # 极高敏感, 全网常驻 (lm_head, ffn.down, 顶层 1-2 层)
    S2 = 1  # 中等敏感, 半随机分布 + 按需流转
    S3 = 2  # 低敏感, 完全自由随机


@dataclass(frozen=True)
class ShardID:
    """分片逻辑身份, 物理位置无关."""

    model_id: str
    layer_index: int
    module_kind: str    # attn.q | attn.kv | ffn.up | ffn.down | embed | lm_head | lora.delta
    tensor_split: int = 0
    quant: str = "fp32"

    def cid(self) -> str:
        s = f"{self.model_id}|{self.layer_index}|{self.module_kind}|{self.tensor_split}|{self.quant}"
        return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()


@dataclass
class ShardMeta:
    """分片元数据, 用于 DHT 寻址与版本对齐."""

    id: ShardID
    blake3_hash: str   # 内容指纹 (此处以 blake2b 代替)
    byte_size: int
    version: int       # 单调递增的迭代步数
    updated_at_unix: float
    sensitivity: Sensitivity
    pvf_percent: int   # 0..100 离线标定


def classify_sensitivity(layer_index: int, module_kind: str, total_layers: int) -> Sensitivity:
    """根据 §4.4.3 的经验规则进行简化分级."""
    if module_kind in {"lm_head", "ffn.down"} or layer_index >= total_layers - 2:
        return Sensitivity.S1
    if module_kind in {"attn.kv", "attn.q", "ffn.up"}:
        return Sensitivity.S2
    return Sensitivity.S3


def shardmeta_to_dict(m: ShardMeta) -> dict:
    d = asdict(m)
    d["id"] = asdict(m.id)
    d["sensitivity"] = int(m.sensitivity)
    return d


def shardmeta_from_dict(d: dict) -> ShardMeta:
    sid = ShardID(**d["id"])
    return ShardMeta(
        id=sid,
        blake3_hash=d["blake3_hash"],
        byte_size=d["byte_size"],
        version=d["version"],
        updated_at_unix=d["updated_at_unix"],
        sensitivity=Sensitivity(d["sensitivity"]),
        pvf_percent=d["pvf_percent"],
    )
