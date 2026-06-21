"""Fused int8 MLP kernel — parity vs the readable per-op reference (the oracle)."""

import numpy as np
import pytest

pytest.importorskip("numba")

from quartz.config import ModelConfig, QuantConfig
from quartz.models import build_model
from quartz.weights import _walk_leaves
from quartz.ops import Linear, RMSNorm, Embedding, silu
from quartz.quantize import quantize_symmetric, quantize_model, QuantizedLinear, FusedMLP
from quartz.kernels.int8 import w8a32_gemv, w8a32_gemv_serial
from quartz.cache import make_cache

TINY = ModelConfig.from_dict(dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, intermediate_size=128, vocab_size=64, rope_theta=10000.0,
    tie_word_embeddings=True, architectures=["Qwen3ForCausalLM"],
))


def _cos(a, b):
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _ql(out, inn, rng, gs=None):
    W = (rng.standard_normal((out, inn)) * 0.05).astype(np.float32)
    q, s = quantize_symmetric(W, gs)
    return QuantizedLinear(q, s, gs, out, inn)


def _fill(net, seed=0):
    rng = np.random.default_rng(seed)
    for _n, m in _walk_leaves(net):
        if isinstance(m, Linear):
            m.weight = (rng.standard_normal((m.out_features, m.in_features)) * 0.05).astype(np.float32)
        elif isinstance(m, RMSNorm):
            m.weight = (1.0 + 0.1 * rng.standard_normal(m.dims)).astype(np.float32)
        elif isinstance(m, Embedding):
            m.weight = (rng.standard_normal((m.num_embeddings, m.dims)) * 0.05).astype(np.float32)
    return net


def test_serial_gemv_bit_identical_to_prange():
    rng = np.random.default_rng(1)
    q, s = quantize_symmetric((rng.standard_normal((200, 128)) * 0.05).astype(np.float32), None)
    x = rng.standard_normal(128).astype(np.float32)
    yp, ys = np.empty(200, np.float32), np.empty(200, np.float32)
    w8a32_gemv(q, s, x, yp)
    w8a32_gemv_serial(q, s, x, ys)
    assert np.array_equal(yp, ys)                      # parallelism is over independent rows


@pytest.mark.parametrize("group_size", [None, 64])
def test_fused_mlp_matches_per_op(group_size):
    rng = np.random.default_rng(0)
    H, INTER = 128, 256
    if group_size and H % group_size:
        pytest.skip("H not divisible by group_size")
    g, u, d = _ql(INTER, H, rng, group_size), _ql(INTER, H, rng, group_size), _ql(H, INTER, rng, group_size)
    fused = FusedMLP(g, u, d)
    x = rng.standard_normal((1, 1, H)).astype(np.float32)       # decode (M=1) -> fused kernel
    ref = d(silu(g(x)) * u(x))
    assert _cos(fused(x), ref) > 0.9999
    x2 = rng.standard_normal((1, 5, H)).astype(np.float32)      # prefill (M>1) -> fallback
    assert np.allclose(fused(x2), d(silu(g(x2)) * u(x2)), atol=1e-4)


def test_fused_model_matches_numpy_int8():
    a = _fill(build_model(TINY))
    b = _fill(build_model(TINY))                       # same seed -> identical fp weights
    quantize_model(a, QuantConfig(default="int8"))
    quantize_model(b, QuantConfig(default="int8", fused=True))
    assert isinstance(b.model.layers[0].mlp, FusedMLP)
    assert isinstance(a.model.layers[0].mlp.gate_proj, QuantizedLinear)

    ids = np.array([[1, 2, 3, 4, 5]])
    assert _cos(a(ids)[0, -1], b(ids)[0, -1]) > 0.9999           # full forward (M>1)

    ca, cb = make_cache(len(a.layers)), make_cache(len(b.layers))
    a.decode_logits(ids, cache=ca)
    b.decode_logits(ids, cache=cb)
    la = a.decode_logits(np.array([[7]]), cache=ca)[0, -1]      # one decode step (M=1, fused kernel)
    lb = b.decode_logits(np.array([[7]]), cache=cb)[0, -1]
    assert _cos(la, lb) > 0.9999


def test_fused_requires_int8_body():
    with pytest.raises(ValueError, match="default='int8'"):
        QuantConfig(default="fp32", fused=True)


@pytest.mark.oracle
def test_fused_real_model_matches_numpy_int8():
    """Fused-body greedy generation must be token-for-token identical to the per-op
    int8 engine on Qwen3-0.6B (same int8 weights; only fp32 reduction order differs)."""
    from quartz.weights import resolve_model_path, load_model
    from quartz.generate import load_tokenizer, generate
    from quartz.config import GenConfig

    try:
        path = resolve_model_path("Qwen/Qwen3-0.6B")
    except Exception as e:  # noqa: BLE001
        pytest.skip(str(e))
    numpy_int8, _ = load_model(path, quant=QuantConfig(default="int8"))
    fused, _ = load_model(path, quant=QuantConfig(default="int8", fused=True))
    tok = load_tokenizer(path)
    cfg = GenConfig(max_tokens=24, temperature=0.0)
    a = generate(numpy_int8, tok, "The capital of France is", cfg, stream=False)
    b = generate(fused, tok, "The capital of France is", cfg, stream=False)
    assert a == b
