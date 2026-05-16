"""任务发起节点临时调度 (架构 §4.4, FR-SCH-01).

任意节点提交任务后晋升为本任务 Coordinator:
    1. RECRUIT 广播招募
    2. 收集 JOIN_ACK 与 SHARD_INVENTORY
    3. 下发 EXEC_PLAN (本 demo: 各 worker 跑相同 mini-batch, 每 N 步聚合)
    4. 监听 HEARTBEAT, 超时 EVICT
    5. 周期收集 GRAD_PUSH, 聚合并广播新版本
    6. TEARDOWN 收尾
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .aggregator import FedAvgOnLoRA, PeerDelta
from .engine import StreamingLoRAEngine
from .gossip import Gossip
from .protocol import Envelope, OpCode

log = logging.getLogger("p2pai.coord")


@dataclass
class TaskSpec:
    task_id: str
    kind: str               # INFERENCE | FINE_TUNE | HYBRID
    model_id: str
    prompts: List[str]
    max_steps: int = 8
    min_group_size: int = 1
    max_group_size: int = 8
    target_p95_latency_ms: int = 5000


@dataclass
class GroupMember:
    peer_id: str
    joined_at: float
    last_heartbeat: float
    pubkey: str


@dataclass
class CoordinatorState:
    spec: TaskSpec
    members: Dict[str, GroupMember] = field(default_factory=dict)
    pending_deltas: List[PeerDelta] = field(default_factory=list)
    finished: bool = False
    results: List[dict] = field(default_factory=list)


class Coordinator:
    """协调器, 每次 submit 实例化一份."""

    HEARTBEAT_TIMEOUT_S = 15.0

    def __init__(self, gossip: Gossip, engine: StreamingLoRAEngine) -> None:
        self.gossip = gossip
        self.engine = engine
        self.aggregator = FedAvgOnLoRA()
        self.tasks: Dict[str, CoordinatorState] = {}

    async def submit(self, spec: TaskSpec) -> CoordinatorState:
        st = CoordinatorState(spec=spec)
        self.tasks[spec.task_id] = st
        log.info("submit task %s kind=%s", spec.task_id, spec.kind)

        # 1. RECRUIT
        env = Envelope(
            op=OpCode.RECRUIT,
            body={
                "task_id": spec.task_id,
                "kind": spec.kind,
                "model_id": spec.model_id,
                "max_steps": spec.max_steps,
                "max_group_size": spec.max_group_size,
            },
            sender_peer="",
            sender_pubkey="",
            task_id=spec.task_id,
        )
        n = await self.gossip.publish(env)
        log.info("RECRUIT broadcast to %d peer(s)", n)

        # 2. 等待 JOIN
        await asyncio.sleep(4.0)
        if len(st.members) < spec.min_group_size:
            log.warning(
                "task %s: only %d/%d members joined; running solo",
                spec.task_id, len(st.members), spec.min_group_size,
            )

        # 3. EXEC_PLAN
        plan_env = Envelope(
            op=OpCode.EXEC_PLAN,
            body={
                "task_id": spec.task_id,
                "prompts": spec.prompts,
                "max_steps": spec.max_steps,
                "aggregate_every": 4,
            },
            sender_peer="",
            sender_pubkey="",
            task_id=spec.task_id,
        )
        await self.gossip.publish(plan_env)

        # 4. 心跳监测
        asyncio.create_task(self._heartbeat_monitor(spec.task_id))
        return st

    def handle_join_ack(self, env: Envelope) -> None:
        st = self.tasks.get(env.task_id or env.body.get("task_id"))
        if not st or st.finished:
            return
        peer = env.sender_peer
        if peer in st.members:
            return
        if len(st.members) >= st.spec.max_group_size:
            return
        st.members[peer] = GroupMember(
            peer_id=peer,
            joined_at=time.time(),
            last_heartbeat=time.time(),
            pubkey=env.sender_pubkey,
        )
        log.info("[%s] member joined: %s (group=%d)",
                 st.spec.task_id[:8], peer[:14], len(st.members))

    def handle_heartbeat(self, env: Envelope) -> None:
        st = self.tasks.get(env.task_id)
        if not st:
            return
        m = st.members.get(env.sender_peer)
        if m:
            m.last_heartbeat = time.time()

    def handle_grad_push(self, env: Envelope) -> None:
        st = self.tasks.get(env.task_id)
        if not st or st.finished:
            return
        body = env.body
        try:
            state = StreamingLoRAEngine.deserialize_state(body["state_b64"])
        except Exception as e:
            log.warning("bad GRAD_PUSH from %s: %s", env.sender_peer, e)
            return
        st.pending_deltas.append(
            PeerDelta(
                peer_id=env.sender_peer,
                version=int(body.get("version", 0)),
                state=state,
                weight=1.0,
            )
        )
        log.info(
            "[%s] received delta from %s (v=%d, pending=%d)",
            st.spec.task_id[:8], env.sender_peer[:14],
            body.get("version", 0), len(st.pending_deltas),
        )

    def handle_infer_result(self, env: Envelope) -> None:
        st = self.tasks.get(env.task_id)
        if not st:
            return
        st.results.append({
            "peer": env.sender_peer,
            "version": env.body.get("version"),
            "output": env.body.get("output"),
            "loss": env.body.get("loss"),
        })

    async def aggregate_and_broadcast(self, task_id: str) -> Optional[int]:
        """协调器侧聚合 LoRA Δ 并向全组广播新版本权重."""
        st = self.tasks.get(task_id)
        if not st or not st.pending_deltas:
            return None
        local = self.engine.export_lora_state()
        new_state = self.aggregator.aggregate(
            local_version=self.engine.version,
            local_state=local,
            peer_deltas=st.pending_deltas,
        )
        self.engine.import_lora_state(new_state)
        st.pending_deltas.clear()
        env = Envelope(
            op=OpCode.GRAD_PUSH,  # 复用消息: 协调器向全组下发聚合后的权重
            body={
                "task_id": task_id,
                "version": self.engine.version,
                "state_b64": StreamingLoRAEngine.serialize_state(new_state),
                "is_aggregated": True,
            },
            sender_peer="",
            sender_pubkey="",
            task_id=task_id,
        )
        await self.gossip.publish(env)
        log.info("[%s] aggregated + broadcast new version v=%d",
                 task_id[:8], self.engine.version)
        return self.engine.version

    async def teardown(self, task_id: str) -> None:
        st = self.tasks.get(task_id)
        if not st:
            return
        st.finished = True
        env = Envelope(
            op=OpCode.TEARDOWN,
            body={"task_id": task_id},
            sender_peer="",
            sender_pubkey="",
            task_id=task_id,
        )
        await self.gossip.publish(env)
        log.info("[%s] task TEARDOWN; final results=%d, members=%d",
                 task_id[:8], len(st.results), len(st.members))

    async def _heartbeat_monitor(self, task_id: str) -> None:
        st = self.tasks.get(task_id)
        if not st:
            return
        while not st.finished:
            await asyncio.sleep(5.0)
            now = time.time()
            evicted = []
            for pid, m in list(st.members.items()):
                if now - m.last_heartbeat > self.HEARTBEAT_TIMEOUT_S:
                    evicted.append(pid)
                    del st.members[pid]
            if evicted:
                log.warning("[%s] EVICT timeout members: %s",
                            task_id[:8], [e[:14] for e in evicted])
