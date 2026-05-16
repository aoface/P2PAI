"""节点身份: Ed25519 PeerID + 消息签名 (架构 §7.6)."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


@dataclass
class Identity:
    """节点身份: PeerID 派生自 Ed25519 公钥的 BLAKE2b-160 哈希."""

    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    @classmethod
    def generate(cls) -> "Identity":
        sk = Ed25519PrivateKey.generate()
        return cls(private_key=sk, public_key=sk.public_key())

    @classmethod
    def load_or_create(cls, path: Path) -> "Identity":
        path = Path(path)
        if path.exists():
            raw = path.read_bytes()
            sk = serialization.load_pem_private_key(raw, password=None)
            return cls(private_key=sk, public_key=sk.public_key())
        path.parent.mkdir(parents=True, exist_ok=True)
        ident = cls.generate()
        pem = ident.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        path.write_bytes(pem)
        path.chmod(0o600)
        return ident

    @property
    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def peer_id(self) -> str:
        """PeerID = base64url(blake2b(pubkey, digest_size=20))."""
        h = hashlib.blake2b(self.public_bytes, digest_size=20).digest()
        return "12D3" + _b64(h)  # 仿 libp2p Ed25519 PeerID 前缀

    def sign(self, payload: bytes) -> str:
        return _b64(self.private_key.sign(payload))


def verify(public_b64: str, payload: bytes, signature_b64: str) -> bool:
    try:
        pk = Ed25519PublicKey.from_public_bytes(_b64d(public_b64))
        pk.verify(_b64d(signature_b64), payload)
        return True
    except Exception:
        return False


def derive_peer_id(public_b64: str) -> str:
    h = hashlib.blake2b(_b64d(public_b64), digest_size=20).digest()
    return "12D3" + _b64(h)


def canonical(payload: dict) -> bytes:
    """规范化 JSON 序列化, 保证签名一致性."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
