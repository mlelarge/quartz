# Fused int8 body — closing (part of) the dispatch wall

M4 found quartz's all-int8 body was **dispatch-bound at ~21 GB/s** (vs ~74 GB/s
bandwidth) — not because int8 math is slow, but because ~196 tiny numba calls +
hundreds of numpy ops per decode step pay too much per-call overhead. This milestone
attacks that. Two fixes, both parity-exact (the per-op path stays the oracle).

All numbers: M3 Max, 12 P-cores, Qwen3-0.6B, batch=1, **same-state interleaved**
(absolutes drift ±15% with load — trust the ratios).

## Fix 1 — don't `prange` small matmuls

Measured: a 1024×1024 int8 GEMV (q/k/v/o-sized, ~1 MB) is **56 µs serial vs 127 µs
with `prange(12)`** — the thread-launch overhead dwarfs the work. `matmul_int8` now
routes sub-2.5 MB int8 matmuls to a **serial** kernel (`w8a32_gemv_serial`, bit-identical
output) and reserves `prange` for the large/stacked ones (lm_head, fused MLP). This one
change lifts **int8-everywhere from ~27 → ~37 tok/s** — i.e. the M4 "anti-result" was
**mostly a threading bug** (prange-on-small-matmuls), not a fundamental int8 cost. With it,
all-int8 is no longer slower than fp32; it's ≈ hybrid.

## Fix 2 — fuse the int8 MLP into one njit call

`FusedMLP` runs `gate + up + silu·up + down` (the whole MLP) in a **single** `@njit(parallel)`
call, with gate|up **stacked** into one `(2·INTER, H)` tensor so a single `prange` amortizes
the launch (gate alone 21 GB/s → stacked gate+up 37 GB/s). It collapses 5
Python↔numba/numpy boundaries (numpy-silu + 3 numba GEMV + numpy-mul) into one. Parity:
`cos = 1.0` vs the per-op reference; greedy generation is **token-for-token identical** to
the numpy-int8 engine on Qwen3-0.6B.

## Result

| config | tok/s | eff GB/s | vs hybrid |
|---|---|---|---|
| fp32 | 29.8 | 71 | 0.82× |
| hybrid (int8 lm_head) | 36.4 | 70 | 1.00× |
| int8 everywhere (serial-dispatch) | 36.9 | 22 | 1.01× |
| **fused int8 body** | **42.7** | **25.5** | **1.17×** |

**The fused int8 body is now the fastest config — 1.17× over hybrid, 1.43× over fp32.**
The cross-engine gap to llama.cpp Q8_0 CPU narrows from ~0.32× to **~0.35–0.38×**
(`docs/results-cross-engine.md`).

## The ceiling — and why we stop here

The fused body lifts effective bandwidth 21 → ~25 GB/s; stacking gives up to ~37 GB/s on
the largest slice. But **llama.cpp hits 74 GB/s** with hand-tuned NEON `sdot` int8, and
that path is **unreachable in numba** — measured: an int8×int8→int32 inner loop (the `sdot`
shape) is **2.3–3× SLOWER**, because LLVM/llvmlite does not lower it to `sdot`/`vpdpbusd`;
the dequant-in-register *fp32-FMA* path is the one that vectorizes. So the body is capped
~25–37 GB/s, the decode ceiling is **~60 tok/s**, and **half of llama.cpp is the honest
limit for a pure numpy+numba engine.** Closing further needs C/intrinsics — which abandons
the transparent-engine thesis. The `w8a8`/VNNI lever is **rejected, with evidence**, so a
future M-pass doesn't re-burn the spike.

## Thesis note

The fused kernel hides 4 ops in one loop — the *opposite* of the transparent per-op design.
We keep the thesis by keeping the **readable per-op MLP as the default and the parity
oracle**: `FusedMLP` is an opt-in accelerator (`QuantConfig(default="int8", fused=True)`)
that must prove bit-equality against the readable version in CI (`tests/test_fused.py`).
Transparency = the readable reference exists and is the source of truth, not that every fast
path is per-op.
