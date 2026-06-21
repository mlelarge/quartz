"""Byte + FLOP roofline model: int8 cuts weight bytes ~4x; the lm_head is ~a third
of the body in BYTES (but a near-equal share of decode TIME — the body is
compute/cache-bound while the lm_head is bandwidth-bound)."""

from quartz.config import ModelConfig
from bench.roofline import byte_budget, weight_bytes_per_token, flops_per_token

CFG = ModelConfig.from_dict(dict(
    hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8, head_dim=128, intermediate_size=3072, vocab_size=151936,
))


def test_int8_cuts_weight_bytes_about_4x():
    fp32 = weight_bytes_per_token(CFG)
    int8 = weight_bytes_per_token(CFG, int8=True, int8_lm_head=True)
    ratio = fp32 / int8
    assert 3.8 < ratio < 4.05               # ~4x (scale overhead is tiny)


def test_lm_head_is_large_share():
    # the tied lm_head/embedding (vocab*hidden) is a large share (~1/3) of the body.
    from bench.roofline import linear_param_count
    c = linear_param_count(CFG)
    assert c["embed_lm_head"] > 0.3 * c["body"]


def test_kv_grows_with_context():
    b0 = byte_budget(CFG, 0)
    b8k = byte_budget(CFG, 8192)
    assert b8k.kv > 0
    assert b0.kv == 0
    assert b8k.total > b0.total


def test_flops_positive_and_grows_with_context():
    f0 = flops_per_token(CFG, 0)
    f8k = flops_per_token(CFG, 8192)
    assert f0 > 0 and f8k > f0
