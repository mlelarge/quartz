# quartz — a transparent, portable CPU LLM inference engine (numpy + Numba)

> Working name **`quartz`** (crystalline SiO₂ = silica's hard, portable crystal form — same
> chemistry as `silica`, a different lattice). Standalone sibling project at
> `/Users/lelarge/Recherche/LLM_Inf/quartz/`. No code dependency on `silica`.

## Context

`silica` is a transparent batch=1 LLM engine for **Apple Silicon / MLX (GPU/Metal)**; its
thesis is *"on a Mac the bottleneck is memory bandwidth, and quantization is the lever."*
`quartz` re-asks that thesis on the **CPU**, as a standalone, **portable (x86 + ARM Linux)**,
**numpy + Numba** engine where **int8 quantization is engineered to be a real decode-speed
lever** — not a pedagogical slowdown.

This direction was validated empirically before committing (M3 Max, numpy 2.4.6 + Accelerate,
numba 0.65.1), reproduced independently by three design agents:

| Finding | Number | Consequence |
|---|---|---|
| fp32 decode is matmul-bound | ~13.3 ms/tok → **~75 tok/s** (lm_head 6.4ms + body 6.9ms) | the fp32 ceiling to beat |
| **fp16 GEMM is 32× SLOWER** | 208 ms vs 6.4 ms | **compute fp32 only**, forever; int8 is a *storage/bandwidth* dtype, not a compute one |
| naive int8 (dequant-to-array) | 10× slower (62 ms dequant) | a **materialization artifact**, not fundamental |
| **fused int8 GEMV (w8a32)** on lm_head | **1.4–1.6 ms, ~100 GB/s, 4.0–4.65×, cos 0.99997** | read int8, **dequant-in-register**, fp32 FMA → DRAM-bandwidth-bound on 4× fewer bytes. The core IP. |
| int8 on small body matmuls | **0.14–0.35×** (BLAS wins) | "int8 everywhere" → 23 tok/s (WORSE). Decision must be **per-matmul**. |
| **hybrid** (fp32 body + int8 lm_head) | **~8.4 ms → ~119 tok/s (1.58×)** | the v0 target |
| measured DRAM ceiling | single `np.sum` **33 GB/s** vs **parallel 121 GB/s** | the "% of peak" denominator must be the **parallel** streaming ceiling, not a single reduction |

**Thesis:** *with a fused int8 kernel, CPU decode becomes DRAM-bandwidth-bound again on the
large memory-bound GEMVs, and quantization recovers as the bandwidth lever — but only where
the matmul is memory-bound; small cache-resident matmuls stay BLAS-bound and must remain fp32.
The per-matmul, size-adaptive dispatch is the engine's central design decision.*

## Goals / non-goals (v0)

- **In:** dense **Qwen3-0.6B** (parity anchor) + **Llama-3.2** attention; fp32 correctness vs an
  independent **HF-transformers-fp32** oracle; the **fused w8a32 int8 GEMV**; a **size-adaptive
  hybrid dispatcher** calibrated on real weights; a re-derived **CPU figure-of-merit** + writeup;
  an **int8 quality (PPL) study**; **portable x86 + ARM Linux** (CI on both).
- **Out (v0):** MoE, quantized/rotating KV, prefix cache, int4 weights, `mx.compile`/async — and
  `silica`'s GPU/Metal paths entirely. (int4 and the x86 VNNI fast-path are M3+ stretch.)
- **Invariants:** fp32 compute only; never materialize a dequantted weight array; attention stays
  numpy/einsum fp32 (cheap: 1.25 ms @ ctx2048). (M1 adds: prefill stays fp32-BLAS, and only the
  **last** prefill token computes lm_head — M0 deliberately projects *all* positions, which the
  teacher-forced parity gate relies on.)

## Architecture

The only genuinely new code is the **kernel layer**; everything numeric is a faithful numpy port
of `silica`'s validated formulas. A model `Linear` calls a dispatcher that, per weight `(N,K)`,
was tagged **once at load** as fp32-BLAS (`x @ W.T`) or fused-int8 — so the hot path never
branches. Platform picks the int8 backend: **`w8a32`** (ARM/Accelerate, validated) vs
**`w8a8`+VNNI** (x86, experimental).

### Project layout
```
quartz/
  pyproject.toml  README.md  LICENSE(MIT)  .github/workflows/ci.yml
  quartz/
    config.py        # VENDOR silica/config.py (trim MLX; QuantConfig→int8; drop kv_bits)
    detokenize.py    # VENDOR silica/detokenize.py VERBATIM (pure-python)
    dtypes.py        # NEW bf16→fp32 upcast (u16.astype(u32)<<16).view(f32); safetensors dtype map
    ops.py           # NEW rmsnorm/silu/softmax/embedding + linear() glue → dispatcher
    rope.py          # NEW half-split RoPE (traditional=False) + llama3 freqs
    attention.py     # NEW GQA einsum SDPA + causal_additive_mask
    cache.py         # NEW NpKVCache (preallocate, (B,n_kv,seq,hd))
    sample.py        # NEW greedy + temp/top-k/top-p/min-p (np.random.default_rng, per-sampler)
    generate.py      # NEW SYNCHRONOUS loop (no async_eval), chat template, stream
    quantize.py      # NEW load-time affine int8 (per-row / per-group g64/g32), QuantizedLinear
    weights.py       # NEW safetensors load (single+sharded) + bf16 upcast + quant policy bind
    kernels/
      __init__.py    # dispatch: pick fp32-BLAS vs int8 per (N,K); JIT warmup
      blas.py        # fp32 x@W.T wrappers + thread pinning (threadpoolctl)
      int8.py        # NEW @njit fused w8a32 + w8a8 GEMV/GEMM — the core IP
      platform.py    # NEW arch/VNNI/BLAS detection; crossover table; NUMBA_THREADING_LAYER
      calibrate.py   # NEW single-pass real-weight crossover calibration
    models/
      __init__.py    # registry (HF architectures → class)
      common.py      # NEW numpy MLP/RMSNorm-call/Decoder/DecoderLayer/CausalLM/tied head
      qwen3.py       # NEW Qwen3Attention (QK-RMSNorm BEFORE RoPE)
      llama.py       # NEW LlamaAttention (no QK-norm, llama3 rope)
  bench/
    roofline.py        # VENDOR silica/bench/roofline.py + int8 byte branch + FLOP/token counter
    streaming_ceiling.py # NEW parallel DRAM-read ceiling (NOT single np.sum)
    decode.py          # NEW tok/s + achieved GB/s + % of measured ceiling, vs context len
    kernel_bench.py    # NEW per-matmul fp32 vs w8a32 vs w8a8, single-pass real weights
  tests/  (see Tests)
```

### The fused kernel (validated)
```python
@njit(parallel=True, fastmath=True, cache=True)   # cache=True needs a real module file
def w8a32_gemv(q, scale, zero, x, y):   # q:(N,K)int8 C-contig; scale/zero:(N,K//G)f32; x:(K,)f32
    N, K = q.shape; ng = scale.shape[1]; G = K // ng
    for n in prange(N):                  # one output row per thread; no reduction race
        acc = np.float32(0.0); qrow = q[n]
        for gi in range(ng):
            s = scale[n,gi]; z = zero[n,gi]; base = gi*G
            for j in range(G):
                acc += (np.float32(qrow[base+j]) * s + z) * x[base+j]   # dequant-in-register
        y[n] = acc
```
- Signed int8, affine group quant **matching `mx.quantize`** (so a silica-MLX 4-bit checkpoint
  is a cross-oracle); zero-point pre-folded (`w ≈ q*scale + zero`). `G∈{64,128}`, `K%G==0`
  enforced. v0 = 8-bit weights (bandwidth thesis); int4 (in-register nibble unpack) deferred.
- Dispatcher tags each Linear at load via the **measured** crossover (`kernel_bench.py`), not a
  hardcoded size. Default profile: **lm_head→int8, body→fp32-BLAS**. Body weights ship int8 on
  disk (4× smaller) and are dequantted to small fp32 arrays **once at load**.

### Numerics carried over from silica (parity-critical — read silica source for exact formulas)
- **RoPE `traditional=False` = HALF-SPLIT (rotate_half), NOT interleaved** — the #1 trap:
  `d=head_dim/2=64`, `inv_freq[j]=base**(-(2j)/D)`, `out[:d]=x1·cos−x2·sin`, `out[d:]=x2·cos+x1·sin`,
  position = `cache.offset`, base=1e6. llama3 freq scaling ports `common._llama3_freqs` (all positions).
- **RMSNorm:** eps **inside** sqrt, on mean-square, no mean-subtraction, eps=1e-6, fp32 accumulate.
- **Qwen3:** per-head QK-RMSNorm over head_dim **before** RoPE; `attention_bias=False`; scale=hd**-0.5.
- **GQA** via `np.repeat(k, n_rep, axis=1)` (contiguous, NOT tile). Causal mask `(seq,offset+seq)`
  fp32 -inf/0, **None when seq≤1**. **Tied lm_head = `h @ embed_W.T`** (this IS the int8 target).
- **bf16 load gotcha (VERIFIED bit-exact):** `safetensors.numpy` raises on bf16 (Qwen3-0.6B is
  bf16) → read raw uint16, `(u16.astype(u32)<<16).view(f32)`. Needs its own test (rewritten, not ported).

## Vendor vs reimplement
- **Copy ~verbatim (already MLX-free):** `silica/detokenize.py`, `silica/config.py` (trim),
  `silica/bench/roofline.py` (+int8 branch +FLOP counter), `silica/bench/eval_ppl.py` algorithm
  (sliding-window NLL), `tests/{conftest,test_config,test_roofline,test_detokenize}.py` patterns.
- **Reimplement in numpy (silica is the numerics oracle, shares no code):** ops, rope, attention,
  cache, sample, generate, weights, models/*. Drop all MLX: `compiled.py`, MoE, async, Metal kernels.

## Figure-of-merit (re-derived for CPU)
- `bench/roofline.py`: byte model + **FLOP/token** counter → arithmetic intensity shows decode is
  far left of the ridge (memory-bound). int8 cuts weight bytes 4×.
- `bench/streaming_ceiling.py`: **parallel** multi-thread streaming read (≥512 MB to defeat LLC)
  → the honest "% of peak" denominator (~121 GB/s here, vs 33 GB/s single-sum, vs the 400 GB/s
  GPU spec the CPU cannot reach — that gap is the GPU-vs-CPU story).
- `bench/decode.py`: headline = **tok/s | achieved GB/s | % of measured ceiling**, vs context len.
- Writeups (silica `results-m2-baseline.md` table style): `docs/results-cpu-roofline.md`
  (GPU-vs-CPU + the crossover table + the hybrid scoreboard + thesis) and
  `docs/results-int8-quality.md` (PPL Pareto, per-layer int8 sensitivity, lm_head cos≈1.0 finding).

## Tests / parity
- **Tier A (pure numpy+Numba, no checkpoint, CI default):** `test_kernels_matmul.py`
  (w8a32/w8a8 cos>0.999 vs fp32 per real shape), `test_rope.py` (**half-split ≠ interleaved
  LOCKED**, offset, llama3 freqs, base=1e6), `test_norm_act.py` (RMSNorm eps-in-sqrt, QK-norm,
  silu/softmax/swiglu), `test_attention_blocks.py` (GQA repeat≠tile, mask, sdpa), `test_cache.py`,
  `test_weights_load.py` (bf16 lossless, tied-head drop), `test_quant.py` (roundtrip, policy).
- **Tier B (`oracle` marker, real checkpoint + HF fp32):** `test_parity.py` — HF-fp32 is the sole
  independent oracle (no MLX): teacher-forced argmax (≥90% interior, last-token exact),
  last-logit cosine (target >0.9999; fall back to 0.999 with documented accumulation-order
  reason), 24-tok greedy match, multibyte no-`�`, Llama-3.2 attention parity. `test_parity_int8.py`
  (argmax-match + top-5 overlap + cos>0.99 vs own fp32 + PPL<1.05×). Optional silica-MLX
  cross-check via its venv (`uv run`).
- A `slow` **timing tripwire**: assert only the crossover *direction* (lm_head int8 faster; small
  MLP fp32 faster), never absolute ms (silica contention lesson).

## Milestones
- **M0 — scaffold + fp32 correctness.** Skeleton, vendored pure-python, numpy runtime (ops/rope/
  attention/cache/model/weights + bf16 upcast), synchronous `generate`, fp32 path only. Gate:
  Tier-A green + HF-fp32 parity on Qwen3-0.6B. Deliverable: `quartz-generate` coherent in pure fp32.
- **M1 — fused int8 + hybrid dispatch.** `kernels/int8.py` (w8a32), `quantize.py`, dispatcher,
  `calibrate.py`. Gate: kernel cos>0.999; int8-path parity; hybrid decode beats fp32 (the 4×
  lm_head, ~119 tok/s). Deliverable: `quartz-kernel-bench` crossover table + hybrid generation.
- **M2 — figure-of-merit + writeups.** streaming ceiling, decode bench, roofline FLOP+byte,
  `docs/results-cpu-roofline.md`, int8 quality PPL study.
- **M3 — portability + x86/VNNI.** `w8a8` kernel + `platform.py` detection; CI matrix
  (ubuntu x86 + ubuntu-24.04-arm); **validate VNNI `vpdpbusd` emission on a real x86 runner**
  (biggest open risk — see below). Gate: ARM-Linux CI green; x86 w8a8 measured (win or documented).
- **M4 (future):** int4 weights, Llama quant quality, prefix cache, MoE.

## Risks (ranked)
1. **w8a8 / VNNI on x86 is a projection, not measured** (dev box is ARM only). Numba/LLVM may not
   emit `vpdpbusd`; mitigate with an `inspect_asm()` CI gate on a real x86 VNNI runner. **w8a32 is
   the guaranteed fallback that already wins 4× on ARM** — v0 ships it as the universal kernel.
2. **In-situ crossover may move** (microbench cache-reuse inflates small-matmul fp32). The
   dispatcher MUST be calibrated single-pass on real weights, not hardcoded — gate via `kernel_bench`.
3. **Numba prange × BLAS thread oversubscription** in the hybrid step. Pin `NUMBA_NUM_THREADS` +
   `OMP/OPENBLAS_NUM_THREADS`; body and lm_head run sequentially → give each all physical cores.
4. **Numba `cache=True` needs a real module file** (fails on REPL) + first-call JIT 0.3–1s → warm
   at import/`load_model`, persist the numba cache dir in CI. `NUMBA_THREADING_LAYER=workqueue`
   is the portable floor (no tbb/omp wheels).
5. fp32 cosine may not reach 0.9999 on the 151936-wide lm_head (accumulation order) — if so that's
   a finding; relax to 0.999 with a documented reason.

## Verification (all runnable locally via `uv run`; no GPU)
```bash
cd /Users/lelarge/Recherche/LLM_Inf/quartz
uv run pytest tests/ -m "not oracle"                                  # Tier A: kernels + numerics
QUARTZ_PARITY_MODEL=Qwen/Qwen3-0.6B uv run pytest tests/test_parity.py -m oracle   # HF fp32 oracle
uv run python -m bench.kernel_bench --model Qwen/Qwen3-0.6B           # crossover table (real weights)
uv run python -m quartz.generate --model Qwen/Qwen3-0.6B --quant lm_head --prompt "Explain RoPE."
uv run python -m bench.decode --model Qwen/Qwen3-0.6B --quant lm_head --bandwidth-gbps <measured>
```
Acceptance: Tier-A green; HF-fp32 parity (argmax + cosine); hybrid `decode` shows the lm_head 4×
and end-to-end ~1.5× over pure fp32; `results-cpu-roofline.md` populated with the scoreboard.
```
```
