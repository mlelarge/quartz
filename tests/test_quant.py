"""Load-time int8 quantization: policy, module swap, and parity vs the fp32 engine."""

import numpy as np
import pytest

pytest.importorskip("numba")

from quartz.config import ModelConfig, QuantConfig
from quartz.models import build_model
from quartz.weights import _walk_leaves, resolve_model_path, load_model
from quartz.ops import Linear, RMSNorm, Embedding
from quartz.quantize import quantize_model, QuantizedEmbedding, QuantizedLinear

TINY = ModelConfig.from_dict(dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, intermediate_size=64, vocab_size=64, rope_theta=10000.0,
    tie_word_embeddings=True, architectures=["Qwen3ForCausalLM"],
))


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


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_default_recipe_quantizes_lm_head_not_body():
    net = _fill(build_model(TINY))
    quantize_model(net, QuantConfig())                  # include = lm_head + embed_tokens
    assert isinstance(net.model.embed_tokens, QuantizedEmbedding)     # tied head -> int8
    assert isinstance(net.model.layers[0].self_attn.q_proj, Linear)   # body stays fp32
    assert isinstance(net.model.layers[0].mlp.down_proj, Linear)


def test_int8_everywhere_swaps_body_too():
    net = _fill(build_model(TINY))
    quantize_model(net, QuantConfig(default="int8"))
    assert isinstance(net.model.layers[0].self_attn.q_proj, QuantizedLinear)
    assert isinstance(net.model.layers[0].mlp.gate_proj, QuantizedLinear)


def test_int8_lm_head_preserves_logits_tiny():
    fp = _fill(build_model(TINY))
    q = _fill(build_model(TINY))
    quantize_model(q, QuantConfig())
    ids = np.array([[1, 2, 3, 4, 5]])
    lf, lq = fp(ids)[0, -1], q(ids)[0, -1]
    assert _cos(lf, lq) > 0.99
    assert int(lf.argmax()) == int(lq.argmax())


@pytest.mark.oracle
def test_int8_parity_real_model():
    """Hybrid int8 lm_head on Qwen3-0.6B stays close to the fp32 engine."""
    try:
        path = resolve_model_path("Qwen/Qwen3-0.6B")
    except Exception as e:  # noqa: BLE001
        pytest.skip(str(e))
    fp, _ = load_model(path)
    q, _ = load_model(path, quant=QuantConfig())
    ids = np.array([[785, 3146, 315, 9625, 374]])       # arbitrary valid token ids
    lf, lq = fp(ids)[0, -1], q(ids)[0, -1]
    assert _cos(lf, lq) > 0.99, _cos(lf, lq)
    top5_f = set(np.argsort(lf)[-5:].tolist())
    top5_q = set(np.argsort(lq)[-5:].tolist())
    assert len(top5_f & top5_q) >= 4
