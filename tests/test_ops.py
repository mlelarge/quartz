"""Numeric primitive locks: RMSNorm (eps inside sqrt), SiLU, softmax."""

import numpy as np

from quartz.ops import rmsnorm, silu, softmax, Linear, Embedding


def test_rmsnorm_formula():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    w = rng.standard_normal(8).astype(np.float32)
    eps = 1e-6
    out = rmsnorm(x, w, eps)
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    exp = x / np.sqrt(ms + eps) * w
    assert np.allclose(out, exp, atol=1e-6)


def test_rmsnorm_eps_is_inside_sqrt():
    """eps added to the mean-square inside the sqrt — NOT outside (a common bug)."""
    x = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    eps = 0.5
    inside = rmsnorm(x, w, eps)
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    outside = x / (np.sqrt(ms) + eps) * w           # the wrong variant
    assert not np.allclose(inside, outside)
    assert np.allclose(inside, x / np.sqrt(ms + eps))


def test_rmsnorm_no_mean_subtraction():
    """RMSNorm normalizes by RMS, not std — a constant-shift input is NOT centered."""
    x = np.array([[5.0, 5.0, 5.0, 5.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    out = rmsnorm(x, w, 1e-6)
    assert np.allclose(out, np.ones(4), atol=1e-3)   # 5/sqrt(25) = 1, not 0


def test_silu():
    x = np.array([-1.0, 0.0, 1.0, 100.0, -100.0], dtype=np.float32)
    out = silu(x)
    ref = x / (1.0 + np.exp(-np.clip(x, -88, 88)))
    assert np.allclose(out, ref)
    assert np.isfinite(out).all()
    assert np.isclose(out[1], 0.0)                   # silu(0) = 0


def test_softmax_sums_to_one_and_handles_neg_inf():
    x = np.array([[1.0, 2.0, -np.inf, 0.5]], dtype=np.float32)
    p = softmax(x, axis=-1)
    assert np.isclose(p.sum(), 1.0)
    assert p[0, 2] == 0.0                            # masked entry -> exactly 0


def test_linear_and_embedding_shapes():
    lin = Linear(8, 5, bias=False)
    lin.weight = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    x = np.random.default_rng(1).standard_normal((2, 3, 8)).astype(np.float32)
    assert lin(x).shape == (2, 3, 5)
    assert np.allclose(lin(x), x @ lin.weight.T)

    emb = Embedding(10, 4)
    emb.weight = np.arange(40, dtype=np.float32).reshape(10, 4)
    ids = np.array([[0, 9]])
    assert np.allclose(emb(ids), emb.weight[ids])
    # tied lm_head: as_linear(x) == x @ weight.T
    h = np.random.default_rng(2).standard_normal((1, 2, 4)).astype(np.float32)
    assert np.allclose(emb.as_linear(h), h @ emb.weight.T)
