"""CLI 入口: 启动节点 / 提交任务."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .node import Node
from .policy import load_config


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def _start_node(config: str, submit: bool, prompts_file: str | None,
                       max_steps: int, min_group: int, run_time: int) -> None:
    cfg = load_config(config)
    node = Node(cfg)
    await node.start()
    # 等待对等节点连接稳定 (引导/HELLO 交换)
    await asyncio.sleep(8.0)
    if submit:
        prompts = _load_prompts(prompts_file)
        task_id = await node.submit_task(
            kind="HYBRID",
            prompts=prompts,
            max_steps=max_steps,
            min_group_size=min_group,
        )
        print(f"[submitted] task_id={task_id}")
    if run_time > 0:
        await asyncio.sleep(run_time)
    else:
        await asyncio.Event().wait()


def _load_prompts(path: str | None) -> list[str]:
    if path:
        return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return [
        "Hello, distributed AI world.",
        "P2P networks enable",
        "The Transformer architecture",
        "Federated learning is",
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="P2PAI demo node")
    p.add_argument("--config", "-c", required=True, help="TOML config path")
    p.add_argument("--submit", action="store_true",
                   help="submit a task after start (this node becomes coordinator)")
    p.add_argument("--prompts", help="path to prompts file")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--min-group", type=int, default=1)
    p.add_argument("--run-time", type=int, default=0,
                   help="exit after N seconds (0 = run forever)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)
    asyncio.run(_start_node(
        args.config, args.submit, args.prompts,
        args.max_steps, args.min_group, args.run_time,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
