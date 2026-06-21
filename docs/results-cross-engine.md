# quartz vs llama.cpp on CPU — the honest external yardstick

Same machine (M3 Max, arm64), same model (Qwen3-0.6B), CPU only. quartz = numpy +
numba; llama.cpp = hand-tuned C++/SIMD (build 9270), run via `llama-bench -ngl 0`.
Each engine at **its best thread count** (see the E-core trap below). `bench/cross_engine.py`.

## The headline

| engine / config | tok/s | threads | bits |
|---|---|---|---|
| quartz fp32 | ~32 | 12 P | fp32 |
| **quartz hybrid (int8 lm_head)** | **~39** | 12 P | mixed |
| quartz int8-everywhere | ~35* | 12 P | int8 |
| **llama.cpp Q8_0** | **~122** | 8 | int8 |
| llama.cpp Q4_K_M | ~163 | 8 | 4-bit (context) |

**quartz's best (hybrid) is ~0.32× llama.cpp Q8_0 on CPU — llama.cpp is ~3.1× faster.**
That is the transparency tax, and the rest of this doc shows *exactly* where it comes from
(it is not where I first thought). (*int8-everywhere tok/s drifts 27–38 with load; its
*bandwidth* below is the stable signal.)

## First, a methodology trap that bit BOTH engines

M3 Max has **12 performance + 4 efficiency cores**. A parallel loop spread across all 16
barrier-stalls on the slow E-cores:

| | -t 6/8 | -t 12 | **-t 16** |
|---|---|---|---|
| llama.cpp Q8_0 CPU | 132 | 128 | **16** |
| quartz int8-everywhere | 38 | 27 | **24** |

A naive "use all cores" run would have reported llama.cpp at **16 tok/s** (slower than
quartz!) — completely wrong. Both must be pinned to the performance cores (`-t 8`;
quartz's `configure_threads()` now defaults to `hw.perflevel0.logicalcpu`). This trap also
**exposed a real quartz bug** (it had been defaulting to all 16) and corrected M2's
int8-everywhere number (24 → ~27).

## The decisive measurement: effective bandwidth

Decode reads the weights once per token, so `bytes/token × tok/s` is the achieved
read bandwidth. The cross-engine numbers line up almost perfectly by **bytes moved**:

| config | tok/s | GB/token | **eff GB/s** |
|---|---|---|---|
| quartz fp32 | 32.0 | 2.384 | **76** |
| quartz hybrid (int8 lm_head) | 38.3 | 1.917 | **74** |
| **llama.cpp Q8_0** | 122.2 | 0.604 | **74** |
| quartz int8-everywhere | 35.0 | 0.596 | **21** ← outlier |

**quartz fp32 and hybrid hit the same ~74–76 GB/s as hand-tuned llama.cpp.** quartz's
kernels (Accelerate fp32 GEMM + the fused int8 lm_head) are **bandwidth-competitive with
C++**. The 3.1× throughput gap is therefore **entirely bytes-moved**: quartz hybrid
streams 1.92 GB/token (its body is fp32) vs llama.cpp's 0.60 GB (all int8) — 3.2× more
bytes, same bandwidth, 3.2× slower.

## Why quartz can't just shrink the bytes

llama.cpp is fast because its **whole body is int8** (0.6 GB/token). quartz's int8-everywhere
config moves the same 0.6 GB — but achieves only **21 GB/s**, not 74. That config is the one
outlier that is *not* bandwidth-bound: it is **dispatch-bound**. Replacing each small body
matmul (q/k/v/o/gate/up/down, ~196 per step) with a numba kernel pays a per-call dispatch +
prange-launch cost that Accelerate's batched GEMM and llama.cpp's fused C loop do not. So
quartz is caught between two regimes:

- **fp32 body** — bandwidth-optimal (74 GB/s) but 3× the bytes → ~39 tok/s.
- **int8 body** — 3× fewer bytes but dispatch-bound (21 GB/s) → ~35 tok/s.

Neither reaches llama.cpp's "few bytes *and* full bandwidth." An efficient all-int8 body at
74 GB/s on 0.6 GB would be **~123 tok/s — llama.cpp parity.** The entire gap is one missing
capability: *a small-matmul int8 GEMM without per-call overhead*, which numpy/numba can't
give but C/SIMD (or a fused-layer kernel) can.

## This corrects M2

M2's `profile_decode` decomposition ("38% matmul / 62% overhead → decode is overhead-bound")
used **isolated, cache-hot** matmul timing, which understated the in-situ cost of the cold
body-DRAM read and so mislabeled bandwidth as "overhead." The cross-engine evidence is
cleaner and overrides it: **decode (fp32/hybrid) is bandwidth-bound** (4× the bytes ⇒ ~3.8×
slower; quartz matches llama.cpp's GB/s). The genuine per-op-overhead problem is real but
**localized to the int8-body path** (the 21 GB/s dispatch wall) — and *that* is what blocks
the bandwidth lever, which is the honest "next lever." (`docs/results-cpu.md` §5 updated.)

## vs silica on the GPU

silica (the MLX/GPU sibling) measured **0.89× llama.cpp** on Metal — a 12% gap. quartz is
**0.32× llama.cpp** on CPU — a 3.1× gap. The CPU transparency tax is far larger, for two
reasons the GPU doesn't have: (1) numpy/numba can't fuse the small int8 body matmuls the way
MLX's `mx.fast` kernels do on-device, so the bandwidth lever (quantization) is stranded; and
(2) there is no `async_eval`-style overlap to hide per-call latency. On the GPU, transparency
costs ~12%; on the CPU, staying in numpy+numba costs ~3×.

## Bottom line

quartz is **bandwidth-competitive with hand-tuned C++** (same ~74 GB/s) and **~3× slower
only because it can't run an efficient all-int8 body** — a single, precisely-located cost of
the numpy+numba design, not a kernel-quality deficit. That is exactly what a transparent
engine should be able to tell you: not just *how far behind*, but *which one missing thing*
accounts for all of it.
