"""Rotary position embeddings (RoPE) — the #1 parity trap.

silica/MLX use `nn.RoPE(traditional=False)`, which is the HALF-SPLIT (rotate_half /
GPT-NeoX / HF-Llama) layout, NOT the interleaved consecutive-pairs layout. The two
preserve the per-vector norm but give DIFFERENT logits, so getting this wrong is a
silent argmax-flipping parity failure.

Layout (head_dim D, half d = D//2):
    inv_freq[j] = base ** (-(2j)/D),   j = 0..d-1
    angle[p, j] = (offset + t) * inv_freq[j]
    x1 = x[..., :d];  x2 = x[..., d:]
    out[..., :d] = x1*cos - x2*sin
    out[..., d:] = x2*cos + x1*sin
This equals HF's `q*cos + rotate_half(q)*sin` with cos/sin of shape (seq, d).

llama3 scaling rescales the *frequencies* (all positions, so parity-critical even
at short context), mirroring silica's `_llama3_freqs`.
"""

from __future__ import annotations

import numpy as np

from .config import ModelConfig


def _llama3_freqs(dims: int, base: float, scaling: dict) -> np.ndarray:
    """Per-pair frequency DENOMINATORS for llama3 RoPE scaling (angle = pos/freq).

    Mirrors silica.models.common._llama3_freqs exactly.
    """
    factor = scaling["factor"]
    low = scaling.get("low_freq_factor", 1.0)
    high = scaling.get("high_freq_factor", 4.0)
    old = scaling.get("original_max_position_embeddings", 8192)
    low_wl = old / low
    high_wl = old / high
    freqs = base ** (np.arange(0, dims, 2, dtype=np.float32) / dims)
    wavelens = 2 * np.pi * freqs
    freqs = np.where(wavelens > low_wl, freqs * factor, freqs)
    is_medium = (wavelens > high_wl) & (wavelens < low_wl)
    smooth = (old / wavelens - low) / (high - low)
    smooth_freqs = freqs / ((1 - smooth) / factor + smooth)
    return np.where(is_medium, smooth_freqs, freqs).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Half-split rotation. x: (..., seq, D); cos/sin: (seq, d=D//2)."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    # broadcast cos/sin over leading (batch, head) axes
    extra = x.ndim - 2
    c = cos.reshape((1,) * extra + cos.shape)
    s = sin.reshape((1,) * extra + sin.shape)
    out = np.empty_like(x)
    out[..., :d] = x1 * c - x2 * s
    out[..., d:] = x2 * c + x1 * s
    return out


class Rope:
    """Callable `rope(x, offset=0)` for x shaped (..., seq, head_dim)."""

    def __init__(self, head_dim: int, base: float, scaling: dict | None = None):
        self.head_dim = head_dim
        if scaling and scaling.get("rope_type") == "llama3":
            freqs_denom = _llama3_freqs(head_dim, base, scaling)
            self.inv_freq = (1.0 / freqs_denom).astype(np.float32)
        else:
            self.inv_freq = (base ** (-(np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
                             ).astype(np.float32)

    def __call__(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        seq = x.shape[-2]
        pos = np.arange(offset, offset + seq, dtype=np.float32)
        angle = pos[:, None] * self.inv_freq[None, :]    # (seq, d)
        return apply_rope(x, np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32))


def build_rope(cfg: ModelConfig) -> Rope:
    return Rope(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
