"""Per-matmul crossover bench: fp32 BLAS vs fused int8 (w8a32), real weights.

Times each Linear/Embedding projection of a real model in ISOLATION at batch=1.
The fused int8 kernel wins big on the large memory-bound lm_head and LOSES on the
small body matmuls (BLAS wins) — the size-adaptive hybrid's whole justification.

CAVEAT (the silica benchmark-contention lesson): timing one matmul in a tight
loop keeps its weight hot in cache, so the small-matmul fp32 numbers here are
cache-INFLATED. The honest end-to-end crossover is in bench/decode.py (a single
streaming pass where each weight is read once per token). This bench shows the
per-kernel ceiling; decode.py shows what survives in situ.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from quartz.weights import load_model, _walk_leaves
from quartz.ops import Linear, Embedding
from quartz.quantize import quantize_symmetric
from quartz.kernels.int8 import w8a32_gemv, warmup


def _median(fn, runs):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser(description="per-matmul fp32 vs fused-int8 crossover (real weights)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    warmup()
    model, _ = load_model(args.model)             # fp32
    rng = np.random.default_rng(0)

    seen = set()
    print(f"{'matmul':<26}{'shape (N,K)':<16}{'fp32 ms':>9}{'int8 ms':>9}{'speedup':>9}{'cos':>9}")
    for name, m in _walk_leaves(model):
        if not isinstance(m, (Linear, Embedding)):
            continue
        W = np.ascontiguousarray(m.weight, dtype=np.float32)
        N, K = W.shape
        kind = name.split(".")[-1]
        key = (kind, N, K)
        if key in seen:                           # one representative per projection type
            continue
        seen.add(key)
        x = rng.standard_normal(K).astype(np.float32)
        x2 = x.reshape(1, K)
        ref = (x2 @ W.T).ravel()

        q, scale = quantize_symmetric(W, args.group_size)
        y = np.empty(N, np.float32)
        w8a32_gemv(q, scale, x, y)                 # warm this shape
        cos = float(y @ ref / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-9))

        fp32_ms = _median(lambda: x2 @ W.T, args.runs) * 1e3
        int8_ms = _median(lambda: w8a32_gemv(q, scale, x, y), args.runs) * 1e3
        print(f"{kind:<26}{f'({N},{K})':<16}{fp32_ms:>9.3f}{int8_ms:>9.3f}"
              f"{fp32_ms / int8_ms:>8.2f}x{cos:>9.4f}")


if __name__ == "__main__":
    main()
