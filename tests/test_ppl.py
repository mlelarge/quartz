"""Perplexity math (pure numpy, no model)."""

import numpy as np

from bench.eval_ppl import log_softmax, perplexity_from_logits


def test_log_softmax_normalized():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 17))
    p = np.exp(log_softmax(x))
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_perplexity_confident_predictor_near_one():
    V = 50
    targets = np.array([3, 9, 1, 42])
    logits = np.full((4, V), -10.0)
    logits[np.arange(4), targets] = 20.0          # near-certain on the right token
    assert perplexity_from_logits(logits, targets) < 1.01


def test_perplexity_uniform_equals_vocab():
    V = 64
    targets = np.array([0, 1, 2, 3, 4])
    logits = np.zeros((5, V))                       # uniform -> PPL == V
    assert abs(perplexity_from_logits(logits, targets) - V) < 1e-6
