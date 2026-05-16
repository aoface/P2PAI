"""GossipSub-like 广播 + 去重 + TTL (架构 §4.2, §6.1)."""

from __future__ import annotations

import collections
import logging
import time
from typing import Awaitable, Callable, Optional

from .identity import Identity, canonical, verify
from .protocol import Envelope
from .transport import Connection, Transport

log = logging.getLogger("p2pai.gossip")

OnEnvelope = Callable[[Envelope, Connection], Awaitable[None]]


class Gossip:
    """带签名验证 + 去重的应用层广播."""

    SEEN_CAPACITY = 4096

    def __init__(
        self,
        transport: Transport,
        identity: Identity,
        on_envelope: OnEnvelope,
    ) -> None:
        self.transport = transport
        self.identity = identity
        self.on_envelope = on_envelope
        self._seen: collections.OrderedDict[str, float] = collections.OrderedDict()

    def _mark_seen(self, msg_id: str) -> bool:
        """若已见返回 True; 否则记录并返回 False."""
        if msg_id in self._seen:
            return True
        self._seen[msg_id] = time.time()
        if len(self._seen) > self.SEEN_CAPACITY:
            self._seen.popitem(last=False)
        return False

    def sign(self, env: Envelope) -> None:
        env.sender_peer = self.identity.peer_id
        env.sender_pubkey = _b64_pub(self.identity)
        env.signature = self.identity.sign(canonical(env.signing_payload()))

    async def publish(self, env: Envelope, exclude: Optional[str] = None) -> int:
        """签名 (若未签名) 并广播; 自身投递一次."""
        if not env.signature:
            self.sign(env)
        self._mark_seen(env.msg_id)
        await self.on_envelope(env, _SelfLoop())
        return await self.transport.broadcast(env.to_dict(), skip=exclude)

    async def on_transport_message(self, conn: Connection, msg: dict) -> None:
        try:
            env = Envelope.from_dict(msg)
        except (KeyError, TypeError):
            log.debug("malformed message")
            return
        if not verify(env.sender_pubkey, canonical(env.signing_payload()), env.signature):
            log.warning("signature invalid from %s", env.sender_peer)
            return
        if self._mark_seen(env.msg_id):
            return  # 已见, 丢弃
        await self.on_envelope(env, conn)
        env.ttl -= 1
        if env.ttl > 0:
            await self.transport.broadcast(env.to_dict(), skip=conn.remote_addr)


def _b64_pub(identity: Identity) -> str:
    from .identity import _b64  # 同模块私有, 避免循环导入
    return _b64(identity.public_bytes)


class _SelfLoop:
    """伪连接, 用于自投递时的占位."""

    remote_addr = "self"
    peer_id = None
    closed = False
