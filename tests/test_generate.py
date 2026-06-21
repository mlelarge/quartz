"""Generation loop edge cases + the load-time guards (from the M0 verification pass)."""

import numpy as np
import pytest

from quartz.config import ModelConfig, GenConfig
from quartz.models import build_model
from quartz.weights import _walk_leaves, bind
from quartz.ops import Linear, RMSNorm, Embedding
from quartz.generate import generate_step

TINY = ModelConfig.from_dict(dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, intermediate_size=64, vocab_size=50, rope_theta=10000.0,
    tie_word_embeddings=True, architectures=["Qwen3ForCausalLM"],
))


def _fill(net, seed=1):
    rng = np.random.default_rng(seed)
    for _n, m in _walk_leaves(net):
        if isinstance(m, Linear):
            m.weight = (rng.standard_normal((m.out_features, m.in_features)) * 0.05).astype(np.float32)
        elif isinstance(m, RMSNorm):
            m.weight = (1.0 + 0.1 * rng.standard_normal(m.dims)).astype(np.float32)
        elif isinstance(m, Embedding):
            m.weight = (rng.standard_normal((m.num_embeddings, m.dims)) * 0.05).astype(np.float32)
    return net


def test_max_tokens_zero_yields_nothing():
    net = _fill(build_model(TINY))
    assert list(generate_step(net, [1, 2, 3], GenConfig(max_tokens=0), ())) == []


def test_max_tokens_one_and_bound():
    net = _fill(build_model(TINY))
    assert len(list(generate_step(net, [1, 2, 3], GenConfig(max_tokens=1), ()))) == 1
    assert len(list(generate_step(net, [1, 2], GenConfig(max_tokens=5), ()))) == 5


def test_empty_prompt_raises_clearly():
    net = _fill(build_model(TINY))
    with pytest.raises(ValueError, match="empty prompt"):
        list(generate_step(net, [], GenConfig(max_tokens=4), ()))


def test_eos_stops_early():
    net = _fill(build_model(TINY))
    first = next(generate_step(net, [1, 2], GenConfig(max_tokens=10), ()))   # greedy, deterministic
    out = list(generate_step(net, [1, 2], GenConfig(max_tokens=10), (first,)))
    assert out == [first]                                                     # stops right after EOS


def test_missing_required_bias_is_detected():
    cfg = ModelConfig.from_dict(dict(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, vocab_size=50, attention_bias=True,
        architectures=["Qwen3ForCausalLM"],
    ))
    net = build_model(cfg)
    _fill(net)                                  # fills weights only; bias=True Linears keep the sentinel
    with pytest.raises(ValueError, match="unbound"):
        bind(net, {})
