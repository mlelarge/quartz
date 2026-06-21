"""Byte-traffic + FLOP model for the CPU figure of merit (ported from silica).

The decode figure of merit counts *all* bytes read per token:

    bytes/token = weight_bytes + kv_bytes(context) + lm_head_bytes + embed_gather

  * weights: fp32 = 4 bytes/weight; int8 = 1 byte + a per-group fp32 scale
    (negligible: +4/group_size bytes/weight). int8 cuts weight traffic ~4x — the
    bandwidth lever a *fused* int8 kernel exploits (M1).
  * kv grows linearly with context and dominates at long context.

On the CPU, decode arithmetic intensity is ~1 FLOP/byte at batch=1, so the FLOP
counter (added here vs silica) confirms decode is far left of the roofline ridge
(memory-bound). The "% of peak" denominator must be the *measured parallel*
streaming ceiling, not a single np.sum and not the GPU spec (added in M2).
"""

from __future__ import annotations

from dataclasses import dataclass

from quartz.config import ModelConfig


def int8_bytes_per_weight(group_size: int | None) -> float:
    """Effective bytes/weight for symmetric int8 incl. one fp32 scale per group
    (per-row scale == group_size of the input dim, so overhead is ~0)."""
    if group_size is None:
        return 1.0          # one fp32 scale per output row -> +~0 over the row
    return 1.0 + 4.0 / group_size


def linear_param_count(cfg: ModelConfig) -> dict[str, int]:
    """Per-token weight parameter counts. At batch=1 decode reads every dense
    weight once."""
    h = cfg.hidden_size
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    per_layer_attn = h * (nq * hd) + 2 * h * (nkv * hd) + (nq * hd) * h   # q,k,v,o
    per_layer_mlp = 3 * h * cfg.intermediate_size                        # gate,up,down
    body = cfg.num_hidden_layers * (per_layer_attn + per_layer_mlp)
    embed = cfg.vocab_size * h          # also the lm_head when tied
    return {"body": body, "embed_lm_head": embed}


def weight_bytes_per_token(
    cfg: ModelConfig, *, int8: bool = False, group_size: int | None = None,
    int8_lm_head: bool = False,
) -> float:
    """Bytes of weights read per decode token. fp32 by default (CPU compute dtype).

    `int8`/`int8_lm_head` mark which parts are stored int8 (the fused-kernel path).
    The input embedding is a GATHER of one row (~hidden floats), NOT a full read.
    """
    counts = linear_param_count(cfg)
    fp32 = 4.0
    body_bpw = int8_bytes_per_weight(group_size) if int8 else fp32
    head_bpw = int8_bytes_per_weight(group_size) if int8_lm_head else fp32
    body_bytes = counts["body"] * body_bpw
    lm_head_bytes = counts["embed_lm_head"] * head_bpw
    input_embed_bytes = cfg.hidden_size * fp32     # a single-row gather
    return body_bytes + lm_head_bytes + input_embed_bytes


def kv_bytes_per_token(cfg: ModelConfig, context_len: int) -> float:
    """Bytes of fp32 KV cache read per decode token at a given context length."""
    per_token_kv = cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2  # K and V
    return per_token_kv * 4.0 * context_len


def flops_per_token(cfg: ModelConfig, context_len: int) -> float:
    """Approx decode FLOPs/token: 2*MACs for every weight read + attention."""
    counts = linear_param_count(cfg)
    matmul = 2.0 * (counts["body"] + counts["embed_lm_head"])
    # QK^T and softmax@V over the context, per head.
    attn = 2.0 * 2.0 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * context_len
    return matmul + attn


@dataclass
class ByteBudget:
    weights: float
    kv: float
    context_len: int

    @property
    def total(self) -> float:
        return self.weights + self.kv

    def achieved_bandwidth_gbps(self, tok_per_s: float) -> float:
        return self.total * tok_per_s / 1e9

    def pct_of_peak(self, tok_per_s: float, peak_gbps: float) -> float:
        return 100.0 * self.achieved_bandwidth_gbps(tok_per_s) / peak_gbps


def byte_budget(cfg: ModelConfig, context_len: int, *, int8: bool = False,
                group_size: int | None = None, int8_lm_head: bool = False) -> ByteBudget:
    return ByteBudget(
        weights=weight_bytes_per_token(cfg, int8=int8, group_size=group_size,
                                       int8_lm_head=int8_lm_head),
        kv=kv_bytes_per_token(cfg, context_len),
        context_len=context_len,
    )
