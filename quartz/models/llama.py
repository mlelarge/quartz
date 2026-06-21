"""Llama decoder (Llama-3.x, SmolLM2, ...). Same RMSNorm -> GQA -> SwiGLU shape as
Qwen3 but with NO QK-norm, and (for Llama-3.x) llama3 RoPE scaling. The whole
runtime is reused; only this small attention block differs.
"""

from __future__ import annotations

from ..config import ModelConfig
from ..ops import Linear
from ..attention import sdpa
from ..rope import build_rope
from .common import CausalLM


class LlamaAttention:
    def __init__(self, cfg: ModelConfig):
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        bias = cfg.attention_bias                 # False for Llama-3
        self.q_proj = Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)
        # No q_norm/k_norm (the Qwen3-vs-Llama difference). RoPE may be llama3-scaled.
        self.rope = build_rope(cfg)

    def __call__(self, x, mask=None, cache=None):
        b, seq, _ = x.shape
        q = self.q_proj(x).reshape(b, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = sdpa(q, k, v, scale=self.scale, mask=mask, n_rep=self.n_rep)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, seq, -1))


class LlamaForCausalLM(CausalLM):
    attention_cls = LlamaAttention


ARCHITECTURES = ("LlamaForCausalLM",)
