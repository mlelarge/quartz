"""Fused int8 kernel correctness vs the fp32 reference (cos > 0.999 per shape)."""

import numpy as np
import pytest

pytest.importorskip("numba")

from quartz.quantize import quantize_symmetric, dequantize_symmetric
from quartz.kernels.int8 import w8a32_gemv, w8a32_gemm, matmul_int8, warmup

warmup()

# real Qwen3-0.6B projection shapes + a tiny one
SHAPES = [(151936, 1024), (3072, 1024), (1024, 3072), (2048, 1024), (1024, 1024), (64, 32)]


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("group_size", [None, 64])
def test_w8a32_gemv_matches_fp32(N, K, group_size):
    if group_size and K % group_size:
        pytest.skip("K not divisible by group_size")
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    x = rng.standard_normal(K).astype(np.float32)
    ref = (x.reshape(1, K) @ W.T).ravel()
    q, scale = quantize_symmetric(W, group_size)
    y = np.empty(N, np.float32)
    w8a32_gemv(q, scale, x, y)
    assert _cos(y, ref) > 0.999


def test_w8a32_gemm_matches_gemv_rows():
    rng = np.random.default_rng(1)
    N, K, M = 128, 64, 5
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    q, scale = quantize_symmetric(W, None)
    X = rng.standard_normal((M, K)).astype(np.float32)
    Y = np.empty((M, N), np.float32)
    w8a32_gemm(q, scale, X, Y)
    for m in range(M):
        y = np.empty(N, np.float32)
        w8a32_gemv(q, scale, np.ascontiguousarray(X[m]), y)
        assert np.allclose(Y[m], y, atol=1e-4)


def test_matmul_int8_dispatch_and_shape():
    rng = np.random.default_rng(2)
    N, K = 100, 64
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    q, scale = quantize_symmetric(W, None)
    x = rng.standard_normal((1, 3, K)).astype(np.float32)        # M=3 -> GEMM path
    out = matmul_int8(q, scale, x)
    assert out.shape == (1, 3, N)
    ref = x @ dequantize_symmetric(q, scale, K).T
    assert _cos(out.ravel(), ref.ravel()) > 0.999


def test_quantize_roundtrip_error_bounded():
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((40, 64)) * 0.05).astype(np.float32)
    q, scale = quantize_symmetric(W, 32)
    Wd = dequantize_symmetric(q, scale, 32)
    # symmetric int8: |error| <= half a quantization step (per-group amax/127)
    assert np.abs(Wd - W).max() <= (np.abs(W).max() / 127.0) * 1.01
    assert q.dtype == np.int8 and q.flags["C_CONTIGUOUS"]
