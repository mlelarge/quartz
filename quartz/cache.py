"""KV cache (numpy port of silica's growing cache).

Layout matches the attention inputs: (batch, n_kv_heads, seq, head_dim). The
cache pre-allocates in `step`-sized chunks so the sequence dimension grows in
coarse jumps (cheap amortized growth, one realloc per `step` tokens) rather than
a fresh concatenate every token.

v0 ships only the growing fp cache. Quantized / rotating / prefix caches are out
of scope (they were optional even in silica and don't compose with quantized KV).
"""

from __future__ import annotations

import numpy as np


class NpKVCache:
    """Growing per-layer KV cache. One instance per decoder layer."""

    def __init__(self, step: int = 256):
        self.keys: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.offset: int = 0          # logical length (valid tokens)
        self.step = step

    def update_and_fetch(self, keys: np.ndarray, values: np.ndarray):
        """Append this step's keys/values, return the full valid slice."""
        prev = self.offset
        n_new = keys.shape[2]
        need = prev + n_new

        if self.keys is None or need > self.keys.shape[2]:
            b, n_kv, _, hd = keys.shape
            n_steps = (need + self.step - 1) // self.step
            new_len = n_steps * self.step
            k_buf = np.zeros((b, n_kv, new_len, hd), dtype=keys.dtype)
            v_buf = np.zeros((b, n_kv, new_len, values.shape[-1]), dtype=values.dtype)
            if self.keys is not None:
                k_buf[..., :prev, :] = self.keys[..., :prev, :]
                v_buf[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = k_buf, v_buf

        self.keys[..., prev:need, :] = keys
        self.values[..., prev:need, :] = values
        self.offset = need
        return self.keys[..., :need, :], self.values[..., :need, :]


def make_cache(n_layers: int, step: int = 256) -> list[NpKVCache]:
    return [NpKVCache(step=step) for _ in range(n_layers)]
