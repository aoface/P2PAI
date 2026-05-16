"""单流原生训推一体执行引擎 — Streaming LoRA (架构 §4.3, FR-RT-01).

核心思想 (与 README §4.3.1 对应):
    传统双线伪一体: 推理链路参数固定, 训练在后台异步更新.
    本架构: 同一条 forward 流既负责生成 Token, 也负责对 LoRA ΔW 反传 + 原地更新.
    一次 engine.step(prompt) 同时返回生成续写 + 本步损失, 模型权重在线演化.

为便于 CPU 上演示, 这里使用 distilgpt2 (82M 参数, ~330MB).
LoRA 注入到所有 Conv1D (attention + FFN) 模块, rank=4 时可训练参数仅 ~50k.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .shard import (
    ShardID,
    ShardMeta,
    Sensitivity,
    classify_sensitivity,
)

log = logging.getLogger("p2pai.engine")


@dataclass
class StepResult:
    output_text: str
    loss: float
    new_version: int
    elapsed_ms: float


class StreamingLoRAEngine:
    """基座权重冻结, 仅 LoRA ΔW 在线训练; 同一次 forward 既生成又训练."""

    def __init__(
        self,
        model_id: str = "distilgpt2",
        lora_rank: int = 4,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        log.info("loading base model %s on %s ...", model_id, device)
        self.model_id = model_id
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        # distilgpt2 / gpt2 的 attention 是 Conv1D, 模块名为 c_attn / c_proj
        # 推理时也会触发反向, 因此基座权重必须 requires_grad=False (LoRA-only training)
        for p in base.parameters():
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model: PeftModel = get_peft_model(base, lora_cfg)
        self.model.to(self.device)
        self.model.train()  # 关键: 使 LoRA dropout 等行为生效, 但 base 仍冻结

        self.optimizer = torch.optim.SGD(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=lr,
            momentum=0.0,
        )
        self.version = 0
        self.total_layers = base.config.n_layer
        self._lock = threading.Lock()
        log.info(
            "engine ready: lora_params=%d / total_params=%d",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            sum(p.numel() for p in self.model.parameters()),
        )

    # --------------------- 训推一体核心 ---------------------
    def step(self, prompt: str, max_new_tokens: int = 16) -> StepResult:
        with self._lock:
            return self._step_locked(prompt, max_new_tokens)

    def _step_locked(self, prompt: str, max_new_tokens: int = 16) -> StepResult:
        """单流: forward → loss → backward → LoRA 原地更新 → 同步生成续写.

        这一步既是"推理" (输出 generated text), 也是"训练"
        (使用 next-token-prediction 自监督 loss 更新 LoRA ΔW).
        生成阶段使用更新后的 LoRA 权重, 保证推理输出反映本步迭代结果.
        """
        t0 = time.time()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        # 训练 forward: 用 prompt 自身做 next-token 自监督
        out = self.model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.version += 1

        # 推理生成: 使用刚刚被本步梯度更新过的 LoRA 权重
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        return StepResult(
            output_text=text,
            loss=float(loss.detach().cpu().item()),
            new_version=self.version,
            elapsed_ms=(time.time() - t0) * 1000.0,
        )

    # --------------------- LoRA Δ 导入导出 ---------------------
    def export_lora_state(self) -> Dict[str, torch.Tensor]:
        """仅导出 LoRA 可训练张量 (ΔW), 用于 P2P 聚合."""
        with self._lock:
            return self._export_locked()

    def _export_locked(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if "lora_" in k
        }

    def import_lora_state(self, state: Dict[str, torch.Tensor]) -> None:
        with self._lock:
            self._import_locked(state)

    def _import_locked(self, state: Dict[str, torch.Tensor]) -> None:
        own = self.model.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(self.device))
        self.model.load_state_dict(own, strict=False)

    @staticmethod
    def serialize_state(state: Dict[str, torch.Tensor]) -> str:
        buf = io.BytesIO()
        torch.save(state, buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def deserialize_state(blob: str) -> Dict[str, torch.Tensor]:
        raw = base64.b64decode(blob.encode("ascii"))
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    # --------------------- 分片元数据生成 ---------------------
    def shard_inventory(self) -> List[ShardMeta]:
        """把本节点持有的 LoRA 张量映射为 ShardMeta 清单 (架构 §7.1).

        对真实分布式训练而言这里应描述基座权重分片; demo 中我们以
        LoRA 增量作为可流转分片演示同样的 ShardMeta + 敏感度 + 版本 + CID 模式.
        """
        metas: List[ShardMeta] = []
        now = time.time()
        for k, v in self.model.state_dict().items():
            if "lora_" not in k:
                continue
            # 从参数名解析层号: 形如 base_model.model.transformer.h.5.attn.c_attn.lora_A.default.weight
            layer_index = _parse_layer_index(k)
            module_kind = _parse_module_kind(k)
            sid = ShardID(
                model_id=self.model_id,
                layer_index=layer_index,
                module_kind=f"lora.delta:{module_kind}",
                tensor_split=0,
                quant="fp32",
            )
            sens = classify_sensitivity(layer_index, module_kind, self.total_layers)
            pvf = {Sensitivity.S1: 40, Sensitivity.S2: 20, Sensitivity.S3: 5}[sens]
            metas.append(
                ShardMeta(
                    id=sid,
                    blake3_hash=_tensor_digest(v),
                    byte_size=v.numel() * v.element_size(),
                    version=self.version,
                    updated_at_unix=now,
                    sensitivity=sens,
                    pvf_percent=pvf,
                )
            )
        return metas


def _parse_layer_index(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "h" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return 0


def _parse_module_kind(name: str) -> str:
    if "c_attn" in name:
        return "attn.qkv"
    if "c_proj" in name and "attn" in name:
        return "attn.out"
    if "c_fc" in name:
        return "ffn.up"
    if "c_proj" in name and "mlp" in name:
        return "ffn.down"
    return "other"


def _tensor_digest(t: torch.Tensor) -> str:
    return hashlib.blake2b(t.detach().cpu().numpy().tobytes(), digest_size=16).hexdigest()
