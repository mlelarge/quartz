# quartz on CPU — figure of merit, the int8 lever, and where the time goes

All numbers: Apple **M3 Max (arm64, 16 cores, Accelerate BLAS)**, Qwen3-0.6B,
batch=1, numpy 2.4.6 + numba 0.65.1. Decode tok/s drifts ±~15% with machine load,
so the **ratios** (which cancel drift) are the load-bearing claims; absolute tok/s
are representative medians.

## 1. The figure of merit is bandwidth, then overhead

At batch=1 a decode step reads every weight once and does ~1 FLOP per weight byte —
deeply memory-bound (far left of the roofline ridge). So the denominator is memory
bandwidth, and **quantization is the lever** because it shrinks the bytes read. That
is the GPU thesis (silica); the question for the CPU is whether it survives. It does —
quartz's fp32/hybrid decode hits the **same ~74 GB/s as hand-tuned llama.cpp** (§5,
[cross-engine](results-cross-engine.md)) — but quartz can only pull the byte-shrinking lever
on the *large* lm_head: an all-int8 body is **dispatch-bound** in numpy/numba, so the body
stays fp32 and that single missing capability is the entire ~3× gap to llama.cpp.

## 2. The measured streaming ceiling — use the parallel number

The honest "% of peak" denominator is a **measured parallel** streaming read, not a
single-threaded `np.sum` (one core can't saturate the controller) and not the chip
*spec* (only the GPU reaches it):

| read | GB/s |
|---|---|
| single-thread `np.sum` | 33 |
| **parallel (numba prange)** | **~115 (112–122 across runs)** |
| ratio | ~3.4–3.6× |

Reporting against ~33 would inflate every result ~3.5×. `bench/streaming_ceiling.py`.

## 3. GPU vs CPU — the same engine, a starved budget

| | silica (M3 Max **GPU**) | quartz (M3 Max **CPU**) |
|---|---|---|
| usable bandwidth | ~370 GB/s | ~115 GB/s (~3.2× less) |
| Qwen3-0.6B decode | 222 tok/s fp16 · 437 4-bit | ~41 tok/s hybrid int8 (~34 fp32) |
| compute dtype | fp16/bf16 fine | **fp32 only** — fp16 GEMM is 32× slower (Accelerate won't vectorize it) |
| decode overlap | `async_eval` hides dispatch (+50%) | none — numpy is eager (no analog) |

**Not a same-precision comparison** — the GPU runs fp16, the CPU is fp32-only (fp16 is
32× slower here), so the apples-to-apples lever is the **~3.2× usable-bandwidth gap**;
fp16-off and the per-op-overhead floor (§5) are *additional* CPU handicaps.

## 4. The int8 lever — fused, and only where it's memory-bound

**Per-matmul, isolated (`bench/kernel_bench.py`):** the fused `w8a32` kernel (read
int8, dequant in-register, fp32 FMA) wins only on the one matmul too large for cache:

| matmul | shape | fp32 | int8 | speedup | cos |
|---|---|---|---|---|---|
| **lm_head** | 151936×1024 | 6.9 ms | **1.62 ms** | **4.25×** | 1.0000 |
| body (q/k/v/o/gate/up/down) | ≤3072×1024 | 0.02–0.05 ms | 0.12–0.21 ms | 0.08–0.32× | ~1.0 |

**Per-class, in situ (`bench/calibrate.py`)** — a real streaming decode, no
cache-reuse inflation; flips one class to int8 and measures end-to-end tok/s:

| int8 class | vs fp32 | verdict |
|---|---|---|
| **embed_tokens (tied lm_head)** | **1.21×** | int8 WIN |
| q/k/v/o_proj | 0.92–0.96× | fp32 win |
| gate/up/down_proj | 0.97–0.99× | fp32 win / neutral |

Only the lm_head crosses over — exactly the default recipe. **"int8 everywhere" is
the anti-result:** forcing int8 onto the small cache-resident body matmuls drops decode
to ~27 tok/s, *below* fp32's ~34 (and to ~24 if numba is left on all 16 cores — §7). The
dispatch must be per-matmul and size-adaptive.

**End-to-end (`bench/decode.py`):**

| config | tok/s | ms/tok |
|---|---|---|
| fp32 | ~34 | ~29 |
| **hybrid (int8 lm_head)** | **~41** | **~24** → **1.21×** |
| int8 everywhere | ~27 | ~37 (anti-result) |

## 5. Where the time goes — it's bytes, not Python overhead

The kernel is 4.25× but the engine is only ~1.2×. An earlier version of this doc blamed
"per-op overhead"; the cross-engine yardstick
([results-cross-engine.md](results-cross-engine.md)) **overturned that — decode is
bandwidth-bound.** quartz fp32 (2.38 GB/token), hybrid (1.92 GB), and hand-tuned llama.cpp
Q8_0 (0.60 GB) all hit the **same ~74–76 GB/s** effective read bandwidth, so tok/s scales as
1/bytes and the int8 lm_head helps exactly in proportion to the bytes it removes.

A `profile_decode.py` decomposition *appears* to show 38% matmul / 62% "overhead":

| | full step | isolated matmuls | "overhead + rest" |
|---|---|---|---|
| **fp32** | 35.4 ms | 13.5 ms (38%) | 21.9 ms (62%) |
| **hybrid** | 29.3 ms | 7.3 ms (25%) | 22.0 ms |

But the "isolated matmuls" are timed **cache-hot** (small body weights reused in a tight
loop), which badly understates the *cold* DRAM read of the whole 1.76 GB body in a real
single-pass step — so it mislabels bandwidth as "overhead." The effective-bandwidth table is
the corrective: most of that "22 ms" is the body's fp32 DRAM read; real
RMSNorm/RoPE/softmax/dispatch is the minority.

**So the lever is bytes (quantization), exactly the GPU thesis** — and quartz pulls it on the
lm_head but *cannot* on the body, because an all-int8 body is **dispatch**-bound:
int8-everywhere achieves only ~21 GB/s (the one config that is *not* bandwidth-bound), numba
paying per-call overhead on ~196 tiny matmuls/step. That stranded bandwidth lever — not a
generic overhead floor — is the real "next lever": a fused / C-level small-matmul int8 GEMM
would let the body go int8 at full bandwidth and approach llama.cpp (~123 tok/s). The GPU
sidesteps it because `mx.fast` fuses the small matmuls and `async_eval` hides the launches.

## 6. Scoreboard

| config | tok/s | eff GB/s | int8 PPL cost | note |
|---|---|---|---|---|
| fp32 | ~32–34 | ~76 | — | baseline; fp16 banned (32× slower) |
| **hybrid (int8 lm_head)** | **~39–41 (1.2×)** | ~74 | **+0.1%** | the recipe; bandwidth-competitive with llama.cpp |
| int8 everywhere | ~27–35 (0.8×) | **~21** | +0.3% | anti-result — *dispatch*-bound, not low quality |

(PPL: `docs/results-int8-quality.md`. Cross-engine: `docs/results-cross-engine.md`.)
Headline: **decode is bandwidth-bound and quartz matches hand-tuned llama.cpp's ~74 GB/s;
the fused int8 kernel pulls the bandwidth lever on the lm_head, but the body stays fp32 (3×
the bytes) because an all-int8 body is dispatch-bound — which is the whole ~3× gap to
llama.cpp.**

## 7. Benchmark rigor

- Ratios over absolutes (decode tok/s drifts ±15% with load); medians + warmup.
- **Pin numba to PERFORMANCE cores.** M3 Max has 12 P + 4 E cores; a `prange` over all 16
  barrier-stalls on the slow E-cores and *collapses* the many-small-kernel int8 path
  (int8-everywhere: ~38 tok/s at 6–8 threads → ~24 at 16). `configure_threads()` defaults to
  the P-core count (`hw.perflevel0.logicalcpu`). The identical trap hits llama.cpp (`-t 16`
  drops its Q8_0 from 132 to 16 tok/s) — both engines must be benched at their best thread
  count (see [docs/results-cross-engine.md](results-cross-engine.md)).
- The crossover is measured **single-pass in situ** (`calibrate.py`), because isolated
  microbenchmarks keep small weights hot in cache and overstate fp32 there.
- "% of peak" uses the **measured parallel** ceiling (~115 GB/s, run-to-run 112–122),
  never single-`np.sum` (~33) and never the GPU spec.
- `bench/decode.py` reports tok/s + ms/tok as the speed FoM; byte-model bandwidth is
  gated behind `--bandwidth-gbps` because bytes×tok/s is not a speed proxy (int8 moves
  fewer bytes, so a faster int8 config can show a lower achieved GB/s).
