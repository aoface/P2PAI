"""用户资源自治层 — 简化的 TOML 策略加载 (架构 §4.1, §7.2)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class UserPolicy:
    """对应架构 §7.2 UserPolicy proto."""

    max_cpu_percent: int = 50
    max_ram_mb: int = 4096
    max_up_kbps: int = 2048
    max_down_kbps: int = 8192
    allow_training: bool = True
    allow_inference: bool = True
    quiet_hours: List[int] = field(default_factory=list)  # 0..23 之间的小时


@dataclass
class NodeConfig:
    """节点级别配置."""

    name: str
    listen_host: str = "127.0.0.1"
    listen_port: int = 7800
    bootstrap_peers: List[str] = field(default_factory=list)  # host:port
    data_dir: str = ".p2pai-demo"
    model_id: str = "distilgpt2"
    lora_rank: int = 4
    lr: float = 1e-3
    aggregate_interval_steps: int = 4
    policy: UserPolicy = field(default_factory=UserPolicy)


def load_config(path: str | Path) -> NodeConfig:
    p = Path(path)
    raw = tomllib.loads(p.read_text())
    pol = UserPolicy(**raw.pop("policy", {}))
    return NodeConfig(policy=pol, **raw)
