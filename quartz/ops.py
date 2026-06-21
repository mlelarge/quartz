"""Numpy fp32 primitives + the leaf modules (Linear / RMSNorm / Embedding).

All compute is fp32 — on the CPU, fp16 GEMM is ~32x slower (Accelerate does not
vectorize it) and int8 is a storage/bandwidth lever (M1), not a compute dtype.

The leaf modules are plain objects holding numpy arrays (no autograd framework).
Their `.weight`/`.bias` start as None and are filled by `weights.bind`, which
walks HF-style dotted keys — so attribute names here mirror HF exactly.
"""

from __future__ import annotations

import numpy as np


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """y = x @ weight.T (+ bias). `weight` is (out, in), matching nn.Linear / HF."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """y = x / sqrt(mean(x^2) + eps) * weight.

    eps is INSIDE the sqrt, added to the mean-square (not a variance, no mean
    subtraction). fp32 accumulation. Matches mx.fast.rms_norm / HF RMSNorm.
    """
    x = x.astype(np.float32, copy=False)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + eps)) * weight).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), numerically stable (clip the exponent within fp32 range)."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """fp32 softmax with max-subtraction (the `precise=True` path). Additive
    -inf mask entries underflow to exactly 0."""
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# --- leaf modules (param holders) ------------------------------------------- #


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight: np.ndarray | None = None
        self.bias: np.ndarray | None = None if not bias else ...  # set at bind

    def __call__(self, x: np.ndarray) -> np.ndarray:
        b = self.bias if isinstance(self.bias, np.ndarray) else None
        return linear(x, self.weight, b)


class RMSNorm:
    def __init__(self, dims: int, eps: float = 1e-6):
        self.dims = dims
        self.eps = eps
        self.weight: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return rmsnorm(x, self.weight, self.eps)


class Embedding:
    def __init__(self, num_embeddings: int, dims: int):
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.weight: np.ndarray | None = None

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """Row gather: ids (..., ) int -> (..., dims)."""
        return self.weight[ids]

    def as_linear(self, x: np.ndarray) -> np.ndarray:
        """Tied output projection: x @ weight.T -> (..., vocab)."""
        return linear(x, self.weight)
