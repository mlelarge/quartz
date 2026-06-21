"""ModelConfig parsing + the Qwen3 traps."""

import pytest

from quartz.config import ModelConfig, QuantConfig, GenConfig

QWEN3_06B = dict(
    hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8, head_dim=128, intermediate_size=3072, vocab_size=151936,
    rope_theta=1_000_000.0, tie_word_embeddings=True, attention_bias=False,
    architectures=["Qwen3ForCausalLM"],
)


def test_from_dict_and_traps():
    cfg = ModelConfig.from_dict(QWEN3_06B)
    assert cfg.head_dim == 128                       # decoupled, NOT 1024//16=64
    assert cfg.n_rep == 2                            # 16 q heads / 8 kv heads
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.architectures == ("Qwen3ForCausalLM",)
    assert cfg.eos_token_ids == (151645,)


def test_head_dim_derived_when_missing():
    d = dict(QWEN3_06B)
    d["head_dim"] = None                             # e.g. SmolLM2
    cfg = ModelConfig.from_dict(d)
    assert cfg.head_dim == 1024 // 16


def test_gqa_divisibility_validated():
    d = dict(QWEN3_06B, num_attention_heads=16, num_key_value_heads=6)
    with pytest.raises(ValueError):
        ModelConfig.from_dict(d)


def test_quantconfig_defaults():
    q = QuantConfig()
    assert q.bits == 8 and q.default == "fp32"
    assert "lm_head" in q.include
    with pytest.raises(ValueError):
        QuantConfig(bits=4)


def test_genconfig_greedy_default():
    g = GenConfig()
    assert g.temperature == 0.0 and g.max_tokens == 256
