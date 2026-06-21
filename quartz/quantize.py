"""Load-time int8 quantization + the quantized leaf modules.

Symmetric per-group affine: `w ~= q * scale`, q int8 in [-127,127], one fp32 scale
per group of `group_size` input-dim elements (None == per-row, the validated
default). Quantized leaves are duck-typed drop-ins for ops.Linear/ops.Embedding
(same `__call__` / `as_linear` interface), so the model code is untouched — the
size-adaptive policy just swaps selected modules at load.

The default recipe (QuantConfig.include = lm_head + embed_tokens) is the spike's
hybrid winner: the large memory-bound lm_head goes int8 (the ~4x fused-GEMV win),
the body stays fp32-BLAS. `quantize_model` applies the policy.
"""

from __future__ import annotations

import numpy as np

from .config import QuantConfig
from .ops import Linear, Embedding
from .kernels.int8 import matmul_int8, warmup
from .kernels.platform import configure_threads
from .weights import set_module, _walk_leaves


def quantize_symmetric(W: np.ndarray, group_size: int | None = None):
    """Quantize (N, K) fp32 weights -> (q int8 (N,K) C-contig, scale fp32 (N,ng))."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    N, K = W.shape
    G = group_size or K
    if K % G != 0:
        raise ValueError(f"group_size {G} does not divide input dim {K}")
    ng = K // G
    Wg = W.reshape(N, ng, G)
    scale = (np.abs(Wg).max(axis=2) / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0                                   # avoid div-by-zero on all-zero groups
    q = np.clip(np.round(Wg / scale[:, :, None]), -127, 127).astype(np.int8).reshape(N, K)
    return np.ascontiguousarray(q), np.ascontiguousarray(scale)


def dequantize_symmetric(q: np.ndarray, scale: np.ndarray, group_size: int) -> np.ndarray:
    """Reconstruct fp32 weights (..., K) from int8 q + per-group scale (reference / gather)."""
    shp = q.shape
    K = shp[-1]
    ng = scale.shape[-1]
    G = K // ng
    qf = q.astype(np.float32).reshape(*shp[:-1], ng, G)
    return (qf * scale[..., None]).reshape(shp).astype(np.float32)


class QuantizedLinear:
    """int8 drop-in for ops.Linear. weight stored as q (out, in) + per-group scale."""

    def __init__(self, q, scale, group_size, out_features, in_features):
        self.q = q
        self.scale = scale
        self.group_size = group_size or in_features
        self.out_features = out_features
        self.in_features = in_features
        self.weight = q          # so a leaf-walker sees it as "bound"

    def __call__(self, x):
        return matmul_int8(self.q, self.scale, x)


class QuantizedEmbedding:
    """int8 drop-in for ops.Embedding. Gather dequantizes the few touched rows
    (cheap); `as_linear` is the fused int8 GEMV (the tied-lm_head win)."""

    def __init__(self, q, scale, group_size, num_embeddings, dims):
        self.q = q
        self.scale = scale
        self.group_size = group_size or dims
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.weight = q

    def __call__(self, ids):
        ids = np.asarray(ids)
        return dequantize_symmetric(self.q[ids], self.scale[ids], self.group_size)

    def as_linear(self, x):
        return matmul_int8(self.q, self.scale, x)


def _wants_int8(name: str, qcfg: QuantConfig) -> bool:
    if any(name.endswith(s) for s in qcfg.exclude):
        return False
    if any(name.endswith(s) for s in qcfg.include):
        return True
    return qcfg.default == "int8"


def quantize_model(net, qcfg: QuantConfig, *, verbose: bool = False):
    """Replace policy-selected Linear/Embedding leaves with int8 versions in place.

    Norms are never touched (only Linear/Embedding leaves are candidates). A leaf
    whose input dim is not divisible by `group_size` is skipped (kept fp32).
    """
    targets = []
    skipped = []
    for name, m in list(_walk_leaves(net)):
        if not isinstance(m, (Linear, Embedding)):
            continue
        if not _wants_int8(name, qcfg):
            continue
        K = m.weight.shape[-1]
        G = qcfg.group_size or K
        if K % G != 0:
            skipped.append(name)
            continue
        targets.append((name, m))

    for name, m in targets:
        q, scale = quantize_symmetric(m.weight, qcfg.group_size)
        if isinstance(m, Embedding):
            new = QuantizedEmbedding(q, scale, qcfg.group_size, m.num_embeddings, m.dims)
        else:
            new = QuantizedLinear(q, scale, qcfg.group_size, m.out_features, m.in_features)
        set_module(net, name, new)

    if verbose and skipped:
        print(f"[quartz] int8 skipped (group_size): {skipped}")
    configure_threads()                       # pin numba to performance cores (avoid E-core collapse)
    warmup()                                  # JIT the kernels once, off the hot path
    return net
