"""Fused int8 weight kernels (Numba) — the core IP of quartz.

The validated win (M3 Max spike): for a large memory-bound GEMV (lm_head,
151936x1024) the FUSED int8 path is ~4x faster than fp32 BLAS because it reads
int8 weights (4x fewer bytes) and dequantizes IN-REGISTER — never materializing
an fp32 weight array (the materialization is the 10x-slower artifact).

Weights are SYMMETRIC per-group int8: w ~= q * scale, q in [-127, 127], one fp32
`scale` per group of `G = K // ng` consecutive input-dim elements (ng == 1 is
per-row, the validated default). The inner loop accumulates int8*fp32 products
per group in fp32, then weights each group sum by its scale — equivalent to the
spike's `acc = (sum x*q) * scale` for the per-row case.

w8a32 = int8 weight, fp32 activation (no activation quant). It WON on ARM/Accelerate.
The int8-activation + VNNI variant (w8a8) is deferred to M3 (needs a real x86 box).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def w8a32_gemv(q, scale, x, y):
    """Decode GEMV (M=1). q:(N,K) int8 C-contig; scale:(N,ng) f32; x:(K,) f32; y:(N,) f32 out."""
    N, K = q.shape
    ng = scale.shape[1]
    G = K // ng
    for n in prange(N):                       # one output row per thread (no reduction race)
        acc = np.float32(0.0)
        for gi in range(ng):
            gs = np.float32(0.0)
            base = gi * G
            for j in range(G):
                gs += x[base + j] * np.float32(q[n, base + j])   # dequant in-register
            acc += gs * scale[n, gi]
        y[n] = acc


@njit(parallel=True, fastmath=True, cache=True)
def w8a32_gemm(q, scale, X, Y):
    """Prefill GEMM (M>1). q:(N,K) int8; scale:(N,ng) f32; X:(M,K) f32 C-contig; Y:(M,N) f32 out.

    The int8 weight row is read once per output channel and reused across the M
    tokens (arithmetic intensity rises with M), so prefill is not purely
    bandwidth-bound — but it never materializes an fp32 weight.
    """
    N, K = q.shape
    ng = scale.shape[1]
    G = K // ng
    M = X.shape[0]
    for n in prange(N):
        for m in range(M):
            acc = np.float32(0.0)
            for gi in range(ng):
                gs = np.float32(0.0)
                base = gi * G
                for j in range(G):
                    gs += X[m, base + j] * np.float32(q[n, base + j])
                acc += gs * scale[n, gi]
            Y[m, n] = acc


def matmul_int8(q, scale, x):
    """y = x @ dequant(q, scale).T for x of shape (..., K). Picks GEMV (M=1) vs GEMM."""
    lead = x.shape[:-1]
    K = x.shape[-1]
    ng = scale.shape[1]
    if K % ng != 0:                      # groups must tile K (quantize_symmetric guarantees this)
        raise ValueError(f"scale has {ng} groups that do not tile K={K}")
    xm = np.ascontiguousarray(x.reshape(-1, K), dtype=np.float32)
    M = xm.shape[0]
    N = q.shape[0]
    y = np.empty((M, N), dtype=np.float32)
    if M == 1:
        w8a32_gemv(q, scale, xm[0], y[0])
    else:
        w8a32_gemm(q, scale, xm, y)
    return y.reshape(*lead, N)


def warmup() -> None:
    """JIT-compile both kernels once (first call is ~0.3-1s; do it off the hot path)."""
    q = np.ones((2, 4), dtype=np.int8)
    s = np.ones((2, 1), dtype=np.float32)
    w8a32_gemv(q, s, np.ones(4, np.float32), np.zeros(2, np.float32))
    w8a32_gemm(q, s, np.ones((2, 4), np.float32), np.zeros((2, 2), np.float32))
