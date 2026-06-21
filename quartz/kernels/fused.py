"""Fused int8 MLP kernel — one njit call for gate + up + silu·up + down (decode, M=1).

M4 found the int8 body is *dispatch*-bound (~21 GB/s), not bandwidth-bound: ~196 tiny
numba calls + hundreds of numpy ops per decode step pay too much per-call overhead.
Fusing the MLP's three projections (+ the silu·up elementwise) into a SINGLE
`@njit(parallel)` call removes 4–5 Python↔numba boundaries on the hottest block.
Spike: 19 → ~29 GB/s, parity cos = 1.0 vs the per-op reference.

gate and up are STACKED into one (2·INTER, H) int8 tensor so a single `prange` over
2·INTER rows amortizes the thread-launch (the spike: stacking lifts gate 21 → 37 GB/s);
the down matmul is a second `prange` in the same call. The small attention matmuls
(q/k/v/o, ≤2 MB) stay SERIAL (prange's ~150 µs launch dominates sub-2MB work) — see
`int8.matmul_int8`'s size dispatch.

w8a8 (int8×int8→int32, the NEON-`sdot` shape) was measured 2.3–3× SLOWER in numba —
LLVM does not lower the scalar int32 MAC to `sdot`; the dequant-in-register fp32-FMA
path is the one that vectorizes. So ~37 GB/s is the numba body ceiling (llama.cpp's
hand-tuned NEON hits 74); fusion NARROWS the gap, it does not close it.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def fused_mlp_decode(gu_q, gu_s, x, down_q, down_s, abuf, out):
    """out[H] = down( silu(gate(x)) * up(x) ), with `x` already RMSNorm'd.

    gu_q:(2*INTER, H) int8 = [gate; up] stacked; gu_s:(2*INTER, ng).
    down_q:(H, INTER) int8; down_s:(H, ng_d). abuf:(INTER,) scratch; out:(H,).
    Rows n and n+INTER of gu_q are the gate and up weights for intermediate n, read
    in one prange iteration so silu·up never materializes two full INTER arrays.
    """
    II = gu_q.shape[0] // 2
    H = x.shape[0]
    ng = gu_s.shape[1]
    G = H // ng
    for n in prange(II):
        ga = np.float32(0.0)
        ua = np.float32(0.0)
        for gi in range(ng):
            gg = np.float32(0.0)
            uu = np.float32(0.0)
            base = gi * G
            for j in range(G):
                xv = x[base + j]
                gg += xv * np.float32(gu_q[n, base + j])
                uu += xv * np.float32(gu_q[n + II, base + j])
            ga += gg * gu_s[n, gi]
            ua += uu * gu_s[n + II, gi]
        s = ga / (np.float32(1.0) + np.exp(-ga))        # silu(gate)
        abuf[n] = s * ua
    ngd = down_s.shape[1]
    Gd = II // ngd
    for n in prange(H):
        acc = np.float32(0.0)
        for gi in range(ngd):
            dd = np.float32(0.0)
            base = gi * Gd
            for j in range(Gd):
                dd += abuf[base + j] * np.float32(down_q[n, base + j])
            acc += dd * down_s[n, gi]
        out[n] = acc


def warmup_fused() -> None:
    """JIT the fused kernel once (off the hot path)."""
    ii, h = 4, 2
    gu = np.ones((2 * ii, h), np.int8)
    gs = np.ones((2 * ii, 1), np.float32)
    dq = np.ones((h, ii), np.int8)
    ds = np.ones((h, 1), np.float32)
    fused_mlp_decode(gu, gs, np.ones(h, np.float32), dq, ds,
                     np.empty(ii, np.float32), np.empty(h, np.float32))
