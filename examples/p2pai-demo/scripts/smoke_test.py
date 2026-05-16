"""不联网的最小冒烟测试: identity 签名 + 引擎单步训推一体 + 聚合器.

用法:
    cd examples/p2pai-demo && python3 scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2pai_demo.aggregator import FedAvgOnLoRA, PeerDelta
from p2pai_demo.engine import StreamingLoRAEngine
from p2pai_demo.identity import Identity, canonical, verify
from p2pai_demo.protocol import Envelope, OpCode


def test_identity() -> None:
    ident = Identity.generate()
    pid = ident.peer_id
    assert pid.startswith("12D3"), pid
    env = Envelope(
        op=OpCode.HELLO, body={"hi": 1},
        sender_peer=pid, sender_pubkey="",
    )
    from p2pai_demo.gossip import _b64_pub
    env.sender_pubkey = _b64_pub(ident)
    env.signature = ident.sign(canonical(env.signing_payload()))
    assert verify(env.sender_pubkey, canonical(env.signing_payload()), env.signature)
    # 篡改后必须失败
    env.body["hi"] = 2
    assert not verify(env.sender_pubkey, canonical(env.signing_payload()), env.signature)
    print("[ok] identity sign/verify")


def test_engine() -> None:
    print("[..] loading distilgpt2 + LoRA (first run will download ~330MB)")
    eng = StreamingLoRAEngine(model_id="distilgpt2", lora_rank=4, lr=1e-3)
    inventory = eng.shard_inventory()
    print(f"[ok] shard inventory: {len(inventory)} LoRA shards")
    s_breakdown = {}
    for m in inventory:
        s_breakdown.setdefault(m.sensitivity.name, 0)
        s_breakdown[m.sensitivity.name] += 1
    print(f"      sensitivity breakdown: {s_breakdown}")

    prompts = [
        "Hello, distributed AI world.",
        "Streaming LoRA enables",
        "Edge devices share",
    ]
    losses = []
    for i in range(4):
        res = eng.step(prompts[i % len(prompts)], max_new_tokens=8)
        losses.append(res.loss)
        print(f"      step={i} v={res.new_version} loss={res.loss:.4f} "
              f"out={res.output_text!r} {res.elapsed_ms:.0f}ms")
    assert losses[-1] <= losses[0] + 0.5, "loss should not explode"
    print(f"[ok] streaming train+infer: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return eng


def test_aggregator(eng: StreamingLoRAEngine) -> None:
    agg = FedAvgOnLoRA()
    local = eng.export_lora_state()
    # 构造一个假的 peer delta (本地复制 + 加噪声)
    import torch
    fake = {k: v.clone() + 0.01 * torch.randn_like(v) for k, v in local.items()}
    merged = agg.aggregate(
        local_version=eng.version,
        local_state=local,
        peer_deltas=[PeerDelta("peer-x", eng.version, fake, weight=1.0)],
    )
    # 期望: 平均后值应介于两者之间
    for k in local:
        avg = (local[k] + fake[k]) / 2
        assert torch.allclose(merged[k], avg, atol=1e-5), k
    print("[ok] aggregator weighted average")

    # 演示陈旧梯度被丢弃
    merged2 = agg.aggregate(
        local_version=100, local_state=local,
        peer_deltas=[PeerDelta("stale", 10, fake)],  # stale=90 > threshold=16
    )
    for k in local:
        assert torch.allclose(merged2[k], local[k]), k
    print("[ok] aggregator drops stale gradient")


def test_serialize(eng: StreamingLoRAEngine) -> None:
    state = eng.export_lora_state()
    blob = StreamingLoRAEngine.serialize_state(state)
    restored = StreamingLoRAEngine.deserialize_state(blob)
    import torch
    for k in state:
        assert torch.allclose(state[k], restored[k]), k
    print(f"[ok] serialize roundtrip ({len(blob)} bytes base64)")


def main() -> int:
    test_identity()
    eng = test_engine()
    test_aggregator(eng)
    test_serialize(eng)
    print("\n*** all smoke tests passed ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
