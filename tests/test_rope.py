"""RoPE parity locks — the #1 trap.

`traditional=False` is the HALF-SPLIT (rotate_half) layout, NOT the interleaved
consecutive-pairs layout. These tests assert the half-split formula AND that it
differs from the interleaved one, so a regression to the wrong layout fails loudly.
"""

import numpy as np

from quartz.rope import Rope, _llama3_freqs


def _interleaved_rope(x, inv_freq, offset=0):
    """traditional=True reference: rotate CONSECUTIVE pairs (x0,x1),(x2,x3),..."""
    seq = x.shape[-2]
    pos = np.arange(offset, offset + seq, dtype=np.float32)
    angle = pos[:, None] * inv_freq[None, :]     # (seq, D/2)
    cos, sin = np.cos(angle), np.sin(angle)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out


def test_half_split_formula():
    rng = np.random.default_rng(0)
    D = 8
    x = rng.standard_normal((1, 2, 3, D)).astype(np.float32)
    rope = Rope(head_dim=D, base=10000.0)
    out = rope(x, offset=0)

    # manual half-split
    d = D // 2
    pos = np.arange(3, dtype=np.float32)
    angle = pos[:, None] * rope.inv_freq[None, :]
    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    x1, x2 = x[..., :d], x[..., d:]
    exp = np.empty_like(x)
    exp[..., :d] = x1 * c - x2 * s
    exp[..., d:] = x2 * c + x1 * s
    assert np.allclose(out, exp, atol=1e-6)


def test_half_split_differs_from_interleaved():
    rng = np.random.default_rng(1)
    D = 8
    x = rng.standard_normal((1, 1, 4, D)).astype(np.float32)
    rope = Rope(head_dim=D, base=10000.0)
    half = rope(x, offset=0)
    inter = _interleaved_rope(x, rope.inv_freq, offset=0)
    assert not np.allclose(half, inter), "half-split must differ from interleaved RoPE"


def test_rope_norm_preserved_per_pair():
    rng = np.random.default_rng(2)
    D = 16
    x = rng.standard_normal((1, 1, 5, D)).astype(np.float32)
    rope = Rope(head_dim=D, base=1e6)
    out = rope(x, offset=3)
    d = D // 2
    # each (i, i+d) pair is a 2D rotation -> magnitude preserved
    before = x[..., :d] ** 2 + x[..., d:] ** 2
    after = out[..., :d] ** 2 + out[..., d:] ** 2
    assert np.allclose(before, after, atol=1e-4)


def test_rope_offset_consistency():
    """Applying RoPE at offset=k to one token equals the (k)th row of applying
    it at offset=0 to k+1 tokens (positions are absolute)."""
    rng = np.random.default_rng(3)
    D = 8
    rope = Rope(head_dim=D, base=10000.0)
    full = rng.standard_normal((1, 1, 6, D)).astype(np.float32)
    out_full = rope(full, offset=0)
    one = full[:, :, 5:6, :]
    out_one = rope(one, offset=5)
    assert np.allclose(out_one[:, :, 0, :], out_full[:, :, 5, :], atol=1e-5)


def test_inv_freq_base_and_length():
    D = 128
    rope = Rope(head_dim=D, base=1_000_000.0)
    assert rope.inv_freq.shape == (D // 2,)
    # inv_freq[0] == 1.0 ; inv_freq[j] = base ** (-(2j)/D)
    assert np.isclose(rope.inv_freq[0], 1.0)
    assert np.isclose(rope.inv_freq[1], 1_000_000.0 ** (-2.0 / D), rtol=1e-5)


def test_llama3_freqs_regimes():
    scaling = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    freqs = _llama3_freqs(128, 500000.0, scaling)
    assert freqs.shape == (64,)
    assert np.all(np.isfinite(freqs))
    # a plain Rope with llama3 scaling differs from one without
    plain = Rope(head_dim=128, base=500000.0)
    scaled = Rope(head_dim=128, base=500000.0, scaling=scaling)
    assert not np.allclose(plain.inv_freq, scaled.inv_freq)
