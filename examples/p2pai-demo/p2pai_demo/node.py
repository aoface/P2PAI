"""节点守护进程: 把 transport / gossip / engine / coordinator 装配成可运行节点."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .coordinator import Coordinator, TaskSpec
from .engine import StreamingLoRAEngine, StepResult
from .gossip import Gossip, _b64_pub
from .identity import Identity
from .policy import NodeConfig
from .protocol import Envelope, OpCode
from .shard import shardmeta_to_dict
from .transport import Connection, Transport

log = logging.getLogger("p2pai.node")


class Node:
    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        data_dir = Path(cfg.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.identity = Identity.load_or_create(data_dir / "identity.pem")
        self.transport = Transport(
            cfg.listen_host, cfg.listen_port, self._on_transport_message,
            on_connected=self._on_connected,
        )
        self.gossip = Gossip(self.transport, self.identity, self._on_envelope)
        self.engine: Optional[StreamingLoRAEngine] = None  # lazy load on first task
        self.coordinator: Optional[Coordinator] = None
        self.peers: Dict[str, str] = {}     # peer_id -> host:port
        self.known_addrs: set[str] = set(cfg.bootstrap_peers)
        self.active_tasks: Dict[str, dict] = {}  # task_id -> worker state

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        await self.transport.start()
        # 预加载引擎
        self.engine = StreamingLoRAEngine(
            model_id=self.cfg.model_id,
            lora_rank=self.cfg.lora_rank,
            lr=self.cfg.lr,
        )
        self.coordinator = Coordinator(self.gossip, self.engine)
        log.info(
            "node %s up at %s:%d  peer_id=%s",
            self.cfg.name, self.cfg.listen_host, self.cfg.listen_port,
            self.identity.peer_id,
        )
        # 引导拨号 + 后台周期性重连 (适配启动顺序不确定)
        asyncio.create_task(self._dialer_loop())
        # 周期性 ANNOUNCE_PROFILE
        asyncio.create_task(self._announce_loop())

    async def stop(self) -> None:
        await self.transport.stop()

    # ------------------------------------------------------------------
    # 传输回调
    # ------------------------------------------------------------------
    async def _on_connected(self, conn: Connection) -> None:
        hello = Envelope(
            op=OpCode.HELLO,
            body={
                "name": self.cfg.name,
                "listen_host": self.cfg.listen_host,
                "listen_port": self.cfg.listen_port,
                "known_addrs": list(self.known_addrs),
            },
            sender_peer="",
            sender_pubkey="",
        )
        self.gossip.sign(hello)
        await conn.send(hello.to_dict())

    async def _on_transport_message(self, conn: Connection, msg: dict) -> None:
        await self.gossip.on_transport_message(conn, msg)

    # ------------------------------------------------------------------
    # 业务层消息分发
    # ------------------------------------------------------------------
    async def _on_envelope(self, env: Envelope, conn) -> None:
        op = env.op
        if op == OpCode.HELLO:
            await self._handle_hello(env, conn)
        elif op == OpCode.ANNOUNCE_PROFILE:
            self.peers[env.sender_peer] = f"{env.body.get('listen_host')}:{env.body.get('listen_port')}"
        elif op == OpCode.RECRUIT:
            await self._handle_recruit(env)
        elif op == OpCode.JOIN_ACK:
            if self.coordinator:
                self.coordinator.handle_join_ack(env)
        elif op == OpCode.EXEC_PLAN:
            await self._handle_exec_plan(env)
        elif op == OpCode.HEARTBEAT:
            if self.coordinator:
                self.coordinator.handle_heartbeat(env)
        elif op == OpCode.GRAD_PUSH:
            await self._handle_grad_push(env)
        elif op == OpCode.INFER_RESULT:
            if self.coordinator:
                self.coordinator.handle_infer_result(env)
        elif op == OpCode.TEARDOWN:
            self.active_tasks.pop(env.task_id, None)
            log.info("task %s torn down", (env.task_id or "?")[:8])

    async def _handle_hello(self, env: Envelope, conn) -> None:
        host = env.body.get("listen_host")
        port = env.body.get("listen_port")
        if host and port:
            addr = f"{host}:{port}"
            self.peers[env.sender_peer] = addr
            if addr not in self.known_addrs:
                self.known_addrs.add(addr)
                # 主动回拨, 形成双向连接
                if addr != f"{self.cfg.listen_host}:{self.cfg.listen_port}":
                    asyncio.create_task(self.transport.dial(addr))
        for addr in env.body.get("known_addrs", []):
            if addr not in self.known_addrs and addr != f"{self.cfg.listen_host}:{self.cfg.listen_port}":
                self.known_addrs.add(addr)
                asyncio.create_task(self.transport.dial(addr))

    async def _dialer_loop(self) -> None:
        """周期性尝试连接所有 known_addrs, 失败重试 (适配节点错峰启动)."""
        while True:
            self_addr = f"{self.cfg.listen_host}:{self.cfg.listen_port}"
            for addr in list(self.known_addrs):
                if addr == self_addr:
                    continue
                conn = self.transport.connections.get(addr)
                if conn is None or conn.closed:
                    await self.transport.dial(addr)
            await asyncio.sleep(3.0)

    async def _announce_loop(self) -> None:
        while True:
            env = Envelope(
                op=OpCode.ANNOUNCE_PROFILE,
                body={
                    "name": self.cfg.name,
                    "listen_host": self.cfg.listen_host,
                    "listen_port": self.cfg.listen_port,
                    "policy": {
                        "max_cpu_percent": self.cfg.policy.max_cpu_percent,
                        "allow_training": self.cfg.policy.allow_training,
                        "allow_inference": self.cfg.policy.allow_inference,
                    },
                    "shard_count": len(self.engine.shard_inventory()) if self.engine else 0,
                    "version": self.engine.version if self.engine else 0,
                },
                sender_peer="",
                sender_pubkey="",
            )
            await self.gossip.publish(env)
            await asyncio.sleep(10.0)

    # ------------------------------------------------------------------
    # Worker 侧: 响应 RECRUIT / EXEC_PLAN / GRAD_PUSH
    # ------------------------------------------------------------------
    async def _handle_recruit(self, env: Envelope) -> None:
        body = env.body
        task_id = body.get("task_id")
        if not task_id or env.sender_peer == self.identity.peer_id:
            return
        # 检查用户策略
        if not self.cfg.policy.allow_training and body.get("kind") in ("FINE_TUNE", "HYBRID"):
            refuse = Envelope(
                op=OpCode.REFUSE,
                body={"task_id": task_id, "reason": "policy:training_disabled"},
                sender_peer="", sender_pubkey="", task_id=task_id,
            )
            await self.gossip.publish(refuse)
            return
        if body.get("model_id") and self.engine and body["model_id"] != self.engine.model_id:
            refuse = Envelope(
                op=OpCode.REFUSE,
                body={"task_id": task_id, "reason": f"model_mismatch:{body['model_id']}"},
                sender_peer="", sender_pubkey="", task_id=task_id,
            )
            await self.gossip.publish(refuse)
            return

        join = Envelope(
            op=OpCode.JOIN_ACK,
            body={
                "task_id": task_id,
                "shard_count": len(self.engine.shard_inventory()) if self.engine else 0,
                "version": self.engine.version if self.engine else 0,
            },
            sender_peer="", sender_pubkey="", task_id=task_id,
        )
        await self.gossip.publish(join)
        self.active_tasks[task_id] = {
            "coordinator": env.sender_peer,
            "step": 0,
            "started_at": time.time(),
        }
        # 启动心跳
        asyncio.create_task(self._heartbeat_loop(task_id))
        log.info("[%s] joined as worker; coord=%s",
                 task_id[:8], env.sender_peer[:14])

    async def _handle_exec_plan(self, env: Envelope) -> None:
        task_id = env.task_id
        if task_id not in self.active_tasks:
            return
        body = env.body
        prompts: List[str] = body.get("prompts", [])
        max_steps = int(body.get("max_steps", 1))
        agg_every = int(body.get("aggregate_every", 4))

        asyncio.create_task(
            self._run_task_loop(task_id, prompts, max_steps, agg_every,
                                coordinator=env.sender_peer)
        )

    async def _run_task_loop(
        self,
        task_id: str,
        prompts: List[str],
        max_steps: int,
        agg_every: int,
        coordinator: str,
    ) -> None:
        assert self.engine is not None
        loss_history: List[float] = []
        for step in range(max_steps):
            prompt = prompts[step % len(prompts)]
            res: StepResult = await asyncio.get_event_loop().run_in_executor(
                None, self.engine.step, prompt
            )
            loss_history.append(res.loss)
            log.info(
                "[%s][step=%d] loss=%.4f  out=%r",
                task_id[:8], step, res.loss, res.output_text[:48],
            )
            # 把推理结果回传给 Coordinator (FR-OBS / 演示用)
            await self.gossip.publish(Envelope(
                op=OpCode.INFER_RESULT,
                body={
                    "task_id": task_id,
                    "step": step,
                    "output": res.output_text,
                    "loss": res.loss,
                    "version": res.new_version,
                },
                sender_peer="", sender_pubkey="", task_id=task_id,
            ))
            # 每 agg_every 步推送一次 LoRA Δ 给协调器, 由其聚合
            if (step + 1) % agg_every == 0:
                state = self.engine.export_lora_state()
                blob = StreamingLoRAEngine.serialize_state(state)
                await self.gossip.publish(Envelope(
                    op=OpCode.GRAD_PUSH,
                    body={
                        "task_id": task_id,
                        "version": self.engine.version,
                        "state_b64": blob,
                        "is_aggregated": False,
                    },
                    sender_peer="", sender_pubkey="", task_id=task_id,
                ))
        log.info("[%s] worker finished; loss %.4f -> %.4f",
                 task_id[:8], loss_history[0] if loss_history else 0,
                 loss_history[-1] if loss_history else 0)

    async def _heartbeat_loop(self, task_id: str) -> None:
        while task_id in self.active_tasks:
            await asyncio.sleep(5.0)
            env = Envelope(
                op=OpCode.HEARTBEAT,
                body={"task_id": task_id, "version": self.engine.version},
                sender_peer="", sender_pubkey="", task_id=task_id,
            )
            await self.gossip.publish(env)

    async def _handle_grad_push(self, env: Envelope) -> None:
        body = env.body
        task_id = env.task_id
        if not task_id:
            return
        # 协调器侧: 收集 worker 推送
        if self.coordinator and task_id in self.coordinator.tasks and not body.get("is_aggregated"):
            self.coordinator.handle_grad_push(env)
            # 简化: 每收到一份就尝试聚合广播
            await self.coordinator.aggregate_and_broadcast(task_id)
            return
        # Worker 侧: 收到协调器下发的聚合权重
        if body.get("is_aggregated") and task_id in self.active_tasks and self.engine:
            try:
                state = StreamingLoRAEngine.deserialize_state(body["state_b64"])
                self.engine.import_lora_state(state)
                self.engine.version = max(self.engine.version, int(body.get("version", 0)))
                log.info("[%s] imported aggregated weights v=%d",
                         task_id[:8], self.engine.version)
            except Exception as e:
                log.warning("import aggregated state failed: %s", e)

    # ------------------------------------------------------------------
    # 任务提交入口 (用户侧)
    # ------------------------------------------------------------------
    async def submit_task(
        self,
        kind: str,
        prompts: List[str],
        max_steps: int = 8,
        min_group_size: int = 1,
        max_group_size: int = 8,
    ) -> str:
        assert self.coordinator is not None and self.engine is not None
        task_id = uuid.uuid4().hex
        spec = TaskSpec(
            task_id=task_id,
            kind=kind,
            model_id=self.engine.model_id,
            prompts=prompts,
            max_steps=max_steps,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
        )
        await self.coordinator.submit(spec)
        # 协调器自身也作为 worker 参与训推一体执行
        self.active_tasks[task_id] = {
            "coordinator": self.identity.peer_id,
            "step": 0,
            "started_at": time.time(),
        }
        asyncio.create_task(self._run_task_loop(
            task_id, prompts, max_steps, 4, coordinator=self.identity.peer_id,
        ))
        return task_id
