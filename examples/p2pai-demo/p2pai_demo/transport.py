"""asyncio TCP 传输层 — 简化版 libp2p (架构 §4.2, §6.1).

特性:
    - 长度前缀 (4 字节 big-endian) + JSON 帧
    - 入向/出向连接, 自动重连
    - 出向连接以 host:port 标识, 入向以 peer_id 标识
    - 与 gossip 层解耦, 仅负责字节传输
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Awaitable, Callable, Dict, Optional

log = logging.getLogger("p2pai.transport")

MAX_FRAME = 4 * 1024 * 1024  # 4 MB, 足够传 distilgpt2 的 LoRA delta

OnMessage = Callable[["Connection", dict], Awaitable[None]]


class Connection:
    """单条 TCP 双向连接的封装."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        remote_addr: str,
        outbound: bool,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.remote_addr = remote_addr  # host:port
        self.outbound = outbound
        self.peer_id: Optional[str] = None  # 握手后填充
        self._send_lock = asyncio.Lock()
        self.closed = False

    async def send(self, msg: dict) -> None:
        data = json.dumps(msg).encode("utf-8")
        if len(data) > MAX_FRAME:
            raise ValueError(f"frame too large: {len(data)}")
        async with self._send_lock:
            try:
                self.writer.write(struct.pack(">I", len(data)) + data)
                await self.writer.drain()
            except (ConnectionError, OSError) as e:
                log.warning("send to %s failed: %s", self.remote_addr, e)
                await self.close()

    async def recv(self) -> Optional[dict]:
        try:
            hdr = await self.reader.readexactly(4)
            (n,) = struct.unpack(">I", hdr)
            if n > MAX_FRAME:
                raise ValueError(f"frame too large: {n}")
            data = await self.reader.readexactly(n)
            return json.loads(data.decode("utf-8"))
        except asyncio.IncompleteReadError:
            return None
        except (ConnectionError, OSError, json.JSONDecodeError) as e:
            log.warning("recv from %s failed: %s", self.remote_addr, e)
            return None

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass


class Transport:
    """节点的 TCP 服务端 + 出向连接管理."""

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        on_message: OnMessage,
        on_connected: Optional[Callable[[Connection], Awaitable[None]]] = None,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.on_message = on_message
        self.on_connected = on_connected
        self.connections: Dict[str, Connection] = {}  # remote_addr -> conn
        self._server: Optional[asyncio.AbstractServer] = None
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_inbound, self.listen_host, self.listen_port
        )
        log.info("listening on %s:%d", self.listen_host, self.listen_port)

    async def dial(self, addr: str) -> Optional[Connection]:
        """addr 为 'host:port' 格式."""
        if addr in self.connections and not self.connections[addr].closed:
            return self.connections[addr]
        host, port_s = addr.rsplit(":", 1)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(port_s)), timeout=3.0
            )
        except (OSError, asyncio.TimeoutError) as e:
            log.debug("dial %s failed: %s", addr, e)
            return None
        conn = Connection(reader, writer, addr, outbound=True)
        self.connections[addr] = conn
        self._spawn(self._read_loop(conn))
        if self.on_connected:
            await self.on_connected(conn)
        return conn

    async def _handle_inbound(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        addr = f"{peer[0]}:{peer[1]}" if peer else "?"
        conn = Connection(reader, writer, addr, outbound=False)
        self.connections[addr] = conn
        if self.on_connected:
            await self.on_connected(conn)
        await self._read_loop(conn)

    async def _read_loop(self, conn: Connection) -> None:
        try:
            while not conn.closed:
                msg = await conn.recv()
                if msg is None:
                    break
                try:
                    await self.on_message(conn, msg)
                except Exception:
                    log.exception("on_message error")
        finally:
            await conn.close()
            self.connections.pop(conn.remote_addr, None)

    def _spawn(self, coro: Awaitable) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def broadcast(self, msg: dict, skip: Optional[str] = None) -> int:
        sent = 0
        for addr, conn in list(self.connections.items()):
            if addr == skip or conn.closed:
                continue
            await conn.send(msg)
            sent += 1
        return sent

    async def stop(self) -> None:
        for conn in list(self.connections.values()):
            await conn.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
