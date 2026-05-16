"""D-PSGD 风格的 LoRA ΔW 加权聚合 (架构 §4.3.3, §7.5).

策略 (FedAvg-on-LoRA):
    - 收集本节点 + 邻居推送的 LoRA state_dict
    - 按节点稳定性评分 (默认 1.0) 加权
    - 引入 staleness 衰减: weight *= 1 / (1 + abs(local_version - peer_version))
    - 超阈值的陈旧梯度直接丢弃
    - 中位数/Krum 防毒化留作可插拔 hook (此处用简单加权平均)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch

log = logging.getLogger("p2pai.aggregator")


@dataclass
class PeerDelta:
    peer_id: str
    version: int
    state: Dict[str, torch.Tensor]
    weight: float = 1.0


class FedAvgOnLoRA:
    def __init__(self, stale_threshold: int = 16) -> None:
        self.stale_threshold = stale_threshold

    def aggregate(
        self,
        local_version: int,
        local_state: Dict[str, torch.Tensor],
        peer_deltas: List[PeerDelta],
    ) -> Dict[str, torch.Tensor]:
        all_deltas = [PeerDelta("self", local_version, local_state, weight=1.0)]
        for pd in peer_deltas:
            stale = abs(local_version - pd.version)
            if stale > self.stale_threshold:
                log.info("drop stale delta from %s (stale=%d)", pd.peer_id, stale)
                continue
            decay = 1.0 / (1.0 + stale)
            all_deltas.append(
                PeerDelta(pd.peer_id, pd.version, pd.state, pd.weight * decay)
            )

        out: Dict[str, torch.Tensor] = {}
        total_w: Dict[str, float] = {}
        for pd in all_deltas:
            for k, v in pd.state.items():
                if k not in out:
                    out[k] = torch.zeros_like(v)
                    total_w[k] = 0.0
                out[k].add_(v * pd.weight)
                total_w[k] += pd.weight
        for k in out:
            if total_w[k] > 0:
                out[k].div_(total_w[k])
        log.info("aggregated %d delta(s)", len(all_deltas))
        return out
