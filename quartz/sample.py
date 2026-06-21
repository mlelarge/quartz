"""Token sampling (numpy). Greedy is the parity-tested path; temperature/top-k/
top-p/min-p are ported for completeness (RNG differs from MLX, so they are not
parity-asserted). RNG is per-sampler (`np.random.default_rng`), not global —
a fixed seed makes one run reproducible without clobbering global state.
"""

from __future__ import annotations

import numpy as np

from .config import GenConfig
from .ops import softmax


def make_sampler(cfg: GenConfig):
    """Return `sampler(logits) -> token_ids` for logits shaped (B, vocab)."""
    greedy = cfg.temperature <= 0.0
    rng = np.random.default_rng(cfg.seed)   # seed None -> fresh entropy

    def sampler(logits: np.ndarray) -> np.ndarray:
        if greedy:
            return np.argmax(logits, axis=-1)

        logits = logits.astype(np.float32) * (1.0 / cfg.temperature)
        if cfg.top_k and cfg.top_k > 0:
            logits = _top_k(logits, cfg.top_k)
        if cfg.min_p and cfg.min_p > 0.0:
            logits = _min_p(logits, cfg.min_p)
        if cfg.top_p and cfg.top_p < 1.0:
            logits = _top_p(logits, cfg.top_p)
        # Gumbel-max: argmax(logits + Gumbel) ~ categorical(softmax(logits)).
        g = rng.gumbel(size=logits.shape).astype(np.float32)
        return np.argmax(logits + g, axis=-1)

    return sampler


def _top_k(logits: np.ndarray, k: int) -> np.ndarray:
    k = min(k, logits.shape[-1])
    kth = np.sort(logits, axis=-1)[..., -k][..., None]
    return np.where(logits < kth, -np.inf, logits)


def _min_p(logits: np.ndarray, min_p: float) -> np.ndarray:
    probs = softmax(logits, axis=-1)
    top = np.max(probs, axis=-1, keepdims=True)
    return np.where(probs < min_p * top, -np.inf, logits)


def _top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    idx = np.argsort(logits, axis=-1)                  # ascending
    sl = np.take_along_axis(logits, idx, axis=-1)
    cum = np.cumsum(softmax(sl, axis=-1), axis=-1)
    keep = cum > (1.0 - top_p)
    masked = np.where(keep, sl, -np.inf)
    inv = np.argsort(idx, axis=-1)
    return np.take_along_axis(masked, inv, axis=-1)
