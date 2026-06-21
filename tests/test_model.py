"""Model self-consistency on a tiny random Qwen3 (no checkpoint needed).

The key invariant: incremental decode through the KV cache (one token at a time)
must reproduce a single full teacher-forced forward. This jointly exercises the
RoPE offset, the growing cache, and the prefill-mask vs single-token-decode paths
— exactly where parity bugs hide.
"""

import numpy as np

from quartz.config import ModelConfig
from quartz.models import build_model
from quartz.weights import _walk_leaves
from quartz.ops import Linear, RMSNorm, Embedding
from quartz.cache import make_cache

TINY = ModelConfig.from_dict(dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=50,
    rope_theta=10000.0, tie_word_embeddings=True, architectures=["Qwen3ForCausalLM"],
))


def _fill_random(net, seed=0):
    rng = np.random.default_rng(seed)
    for _name, m in _walk_leaves(net):
        if isinstance(m, Linear):
            m.weight = (rng.standard_normal((m.out_features, m.in_features)) * 0.05).astype(np.float32)
        elif isinstance(m, RMSNorm):
            m.weight = (1.0 + 0.1 * rng.standard_normal(m.dims)).astype(np.float32)
        elif isinstance(m, Embedding):
            m.weight = (rng.standard_normal((m.num_embeddings, m.dims)) * 0.05).astype(np.float32)


def test_build_and_forward_shape():
    net = build_model(TINY)
    _fill_random(net)
    ids = np.array([[1, 2, 3, 4]])
    logits = net(ids)
    assert logits.shape == (1, 4, TINY.vocab_size)
    assert np.isfinite(logits).all()


def test_incremental_decode_equals_full_forward():
    net = build_model(TINY)
    _fill_random(net, seed=7)
    ids = [5, 9, 2, 8, 1]

    full = net(np.array([ids]))                        # (1, L, V) no cache

    cache = make_cache(len(net.layers))
    inc = []
    for t in ids:
        out = net(np.array([[t]]), cache=cache)        # one token at a time
        inc.append(out[0, -1])
    inc = np.stack(inc)                                 # (L, V)

    assert np.allclose(full[0], inc, atol=1e-4), \
        f"max abs diff {np.abs(full[0] - inc).max():.2e}"


def test_unbound_parameters_raise():
    import pytest
    from quartz.weights import bind
    net = build_model(TINY)
    with pytest.raises(ValueError, match="unbound"):
        bind(net, {})            # nothing bound -> every leaf still None
