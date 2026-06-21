"""Shared layer library (numpy port of silica.models.common).

Plain Python objects holding numpy arrays — no autograd framework. Per-architecture
files (`qwen3.py`, `llama.py`) supply only their attention module; everything
model-agnostic (MLP, decoder stack, tied lm_head) lives here. Attribute names
mirror HF checkpoints so `weights.bind` can walk dotted keys directly.
"""

from __future__ import annotations

import numpy as np

from ..config import ModelConfig
from ..ops import Embedding, Linear, RMSNorm, silu
from ..attention import causal_additive_mask


class MLP:
    """SwiGLU MLP (shared by Qwen3 and Llama)."""

    def __init__(self, cfg: ModelConfig):
        self.gate_proj = Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer:
    def __init__(self, cfg: ModelConfig, attention_cls):
        self.self_attn = attention_cls(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Decoder:
    """Embedding -> N decoder layers -> final norm. Names match HF checkpoints."""

    def __init__(self, cfg: ModelConfig, attention_cls):
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg, attention_cls) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        offset = cache[0].offset if cache[0] is not None else 0
        mask = causal_additive_mask(h.shape[1], offset, h.dtype)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class CausalLM:
    """Decoder-only LM base. Subclasses set `attention_cls`. Handles tied vs
    untied lm_head (tied -> the embedding matrix IS the output projection)."""

    attention_cls = None

    def __init__(self, cfg: ModelConfig):
        self.config = cfg
        self.model = Decoder(cfg, type(self).attention_cls)
        if not cfg.tie_word_embeddings:
            self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        # accept python lists / nested lists of ids -> int array
        inputs = np.asarray(inputs)
        h = self.model(inputs, cache)
        return self._project(h)

    def decode_logits(self, inputs, cache=None):
        """Logits for the LAST position only (B, 1, vocab) — the generation path.

        Both prefill (full prompt -> last token) and decode (one token) need only
        the final position's logits, so we run the lm_head on h[:, -1:, :] alone.
        This avoids the big vocab projection over every prompt token — the entire
        point of int8'ing the lm_head, since that projection is the int8 target."""
        inputs = np.asarray(inputs)
        h = self.model(inputs, cache)[:, -1:, :]
        return self._project(h)

    def _project(self, h):
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        """Hook for HF->quartz weight remapping (no-op for dense models)."""
        return weights
