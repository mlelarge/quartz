"""Typed configuration (ported from silica; pure-Python, no MLX).

`ModelConfig` mirrors the fields read from a HuggingFace `config.json`. The
audit-pinned Qwen3 traps are encoded as explicit, *required* fields rather than
derived quantities:

  * `head_dim` is decoupled from hidden_size/num_heads (Qwen3-0.6B: 128, NOT
    1024/16 = 64). Always read it from config.
  * `attention_bias` is False on Qwen3 (Qwen2 used QKV bias).
  * `tie_word_embeddings` is True on 0.6B (no separate lm_head weight).
  * `rope_theta` is 1e6 (the RoPE default of 1e4 would break parity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    # --- architecture (read straight from config.json) ---
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int                 # decoupled; do NOT compute hidden//heads
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    hidden_act: str = "silu"
    # Long-context RoPE scaling, e.g. {"rope_type": "llama3", "factor": 8.0, ...}.
    # None == native context only (Qwen3-0.6B has none).
    rope_scaling: dict | None = None

    # --- special tokens (Qwen3 defaults) ---
    bos_token_id: int | None = 151643      # <|endoftext|>
    eos_token_id: int | tuple[int, ...] = 151645  # <|im_end|>
    model_type: str = "qwen3"
    # HF `architectures` field -> selects the quartz model class (the registry).
    architectures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})."
            )

    @property
    def n_rep(self) -> int:
        """GQA repeat factor (query heads per kv head)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        e = self.eos_token_id
        return tuple(e) if isinstance(e, (list, tuple)) else (e,)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        d = dict(d)
        # head_dim may be absent OR explicitly null (e.g. SmolLM2); derive it.
        if d.get("head_dim") is None:
            d["head_dim"] = d["hidden_size"] // d["num_attention_heads"]
        if isinstance(d.get("architectures"), list):
            d["architectures"] = tuple(d["architectures"])
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True)
class QuantConfig:
    """Selective int8 quantization policy (used from M1 onward; inert in M0).

    quartz quantizes weights to int8 and runs a *fused* int8 kernel only where a
    matmul is memory-bound (the size-adaptive hybrid). The default recipe keeps
    the body in fp32-BLAS and quantizes the large lm_head — the spike's ~119 tok/s
    winner. Norms are always left fp32.
    """

    # Per-matmul policy by HF-key suffix. Default body precision and an include
    # list of suffixes forced to int8 (lm_head is the prime, lossless-ish target).
    default: str = "fp32"                 # "fp32" | "int8"
    include: tuple[str, ...] = ("lm_head", "embed_tokens")  # tied head == embedding
    exclude: tuple[str, ...] = ()         # forced fp32 (norms are always fp32)
    bits: int = 8                         # 8-bit weight lever (int4 is future work)
    group_size: int | None = None         # None == per-row scale; 64/128 == per-group
                                           # (a near-free quality win: worst logit cos
                                           # 0.99998 vs 0.99997 per-row on Qwen3-0.6B)
    kernel: str = "auto"                  # "auto" | "w8a32" | "w8a8"
    # Fuse each layer's MLP (gate+up+silu+down) into one njit call for decode — the
    # M4 fix for the dispatch-bound int8 body. Requires default="int8" (the whole body
    # must be int8). Opt-in; the per-op MLP stays the readable parity reference.
    fused: bool = False

    def __post_init__(self) -> None:
        if self.bits != 8:
            raise ValueError(f"v0 supports int8 only (bits={self.bits})")
        if self.group_size is not None and self.group_size not in (32, 64, 128):
            raise ValueError(f"unsupported group_size={self.group_size}")
        if self.kernel not in ("auto", "w8a32", "w8a8"):
            raise ValueError(f"unsupported kernel={self.kernel}")
        if self.fused and self.default != "int8":
            raise ValueError("fused=True requires default='int8' (the whole body must be int8)")


@dataclass(frozen=True)
class GenConfig:
    """Sampling + generation controls."""

    max_tokens: int = 256
    temperature: float = 0.0        # 0.0 == greedy
    top_k: int = 0                  # 0 == disabled
    top_p: float = 1.0
    min_p: float = 0.0
    # None -> fresh randomness each generation; an int makes a single run
    # reproducible WITHOUT touching global RNG state (per-sampler default_rng).
    seed: int | None = None
    stop: tuple[str, ...] = ()      # extra string stop-sequences
    # apply the model's chat template before prefill (ChatML for Qwen3).
    use_chat_template: bool = True


@dataclass(frozen=True)
class BenchConfig:
    """Measurement harness knobs. Defaults encode the rigor asks."""

    warmup: int = 3                 # discarded iterations (JIT / pipeline build)
    runs: int = 10                  # K runs for median + IQR
    # The denominator for "% of peak bandwidth" MUST be the *measured* parallel
    # CPU streaming ceiling (not a single np.sum, not the GPU spec). No default
    # that could silently mislabel a result.
    device_bandwidth_gbps: float | None = None
    machine_name: str = "unknown-cpu"
    context_lengths: tuple[int, ...] = field(default=(0, 2048, 8192, 32768))
