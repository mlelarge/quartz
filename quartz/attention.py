"""Scaled-dot-product attention (numpy fp32) + the offset-aware causal mask.

silica dispatches to `mx.fast.scaled_dot_product_attention`; here it is a small
explicit einsum/matmul path. Attention is cheap on the CPU (~1.25 ms/token at
ctx 2048), so it stays fp32 and is never quantized.
"""

from __future__ import annotations

import numpy as np

from .ops import softmax


def causal_additive_mask(seq_len: int, offset: int, dtype=np.float32) -> np.ndarray | None:
    """Additive causal mask, shape (seq_len, offset+seq_len).

    None for single-token decode (a query attends to all cached keys).
    """
    if seq_len <= 1:
        return None
    total = offset + seq_len
    q_pos = np.arange(offset, total).reshape(seq_len, 1)
    k_pos = np.arange(total).reshape(1, total)
    allowed = k_pos <= q_pos
    return np.where(allowed, np.array(0.0, dtype), np.array(-np.inf, dtype))


def sdpa(q, k, v, *, scale, mask, n_rep):
    """GQA + causal attention. q: (B, n_q, Lq, D); k, v: (B, n_kv, Lk, D).

    GQA is handled by repeating each kv head `n_rep` times along the head axis
    with `np.repeat` (contiguous grouping — each kv head feeds n_rep adjacent
    query heads, matching MLX; `np.tile` would interleave wrongly).
    """
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)
    scores = (q @ np.swapaxes(k, -1, -2)) * scale     # (B, n_q, Lq, Lk)
    if mask is not None:
        scores = scores + mask
    p = softmax(scores, axis=-1)
    return p @ v                                       # (B, n_q, Lq, D)
