# quartz

A transparent, **portable CPU** LLM inference engine in **numpy + Numba** — the
CPU sibling of [`silica`](../silica) (which targets Apple Silicon / MLX). Same
transparent, single-stream design; the bottleneck and the levers move.

> **Status: M0–M2 + M4 done.** Dense Qwen3 + Llama decode in fp32 (parity-exact vs an
> independent HuggingFace fp32 oracle, cosine 1.000000); the fused int8 kernel +
> size-adaptive hybrid dispatcher (**4.25×** on the lm_head, ~1.2× end-to-end, int8 PPL
> cost +0.1%); a measured figure-of-merit; an external yardstick vs `llama.cpp` CPU
> (quartz is **bandwidth-competitive (~74 GB/s, == llama.cpp)** but **~3× slower**, from
> moving more bytes); and a **fused int8-body kernel** making all-int8 the fastest config
> (**1.17× over hybrid, 1.43× over fp32**), to the numba ceiling. See
> **[results-fused.md](docs/results-fused.md)**, **[results-cross-engine.md](docs/results-cross-engine.md)**,
> **[results-cpu.md](docs/results-cpu.md)**, **[results-int8-quality.md](docs/results-int8-quality.md)**.

## The thesis

On the GPU (`silica`), decode is memory-bandwidth-bound and **quantization is the
lever**. On the CPU that inverts *naively* — but a measured spike shows it
recovers with the right kernel:

- **fp32 is the compute dtype.** fp16 GEMM is ~32× slower (Accelerate doesn't
  vectorize it); int8 is a *storage/bandwidth* lever, not a compute dtype.
- **A *fused* int8 GEMV** (read int8 weights, dequant **in-register**, fp32 FMA —
  never materialize an fp32 weight array) is **~4× faster** than fp32 BLAS on the
  large, memory-bound lm_head (cos 0.99997), because it is DRAM-bandwidth-bound on
  4× fewer bytes.
- **But int8 loses on small cache-resident matmuls** (BLAS wins), so "int8
  everywhere" is *slower* than fp32. The engine's core decision is a
  **size-adaptive, per-matmul hybrid**: fp32 BLAS for the body, fused-int8 for the
  lm_head — measured **~41 vs ~34 tok/s (1.21×)** end-to-end on an M3 Max.
- **The win is capped by bytes, not Python overhead.** Decode is bandwidth-bound:
  quartz fp32/hybrid hit the *same* **~74 GB/s as hand-tuned llama.cpp** (so the kernels
  are competitive), and the ~1.2× ceiling is that the body stays fp32 (3× the bytes) — an
  all-int8 body is *dispatch*-bound in numpy/numba. Net: quartz is **~0.32× llama.cpp Q8_0
  on CPU (it's ~3.1× faster)**, a gap that is *entirely* bytes-moved. See
  **[docs/results-cross-engine.md](docs/results-cross-engine.md)** and [results-cpu.md](docs/results-cpu.md).

## Layout

```
quartz/
  config.py       typed config (ModelConfig from HF config.json, Quant/Gen/Bench)
  dtypes.py       safetensors load + bf16->fp32 upcast (safetensors.numpy can't read bf16)
  ops.py          fp32 primitives (rmsnorm/silu/softmax/linear) + leaf modules
  rope.py         half-split RoPE (traditional=False) + llama3 freq scaling
  attention.py    GQA einsum SDPA + offset-aware causal mask
  cache.py        growing fp KV cache (preallocate in step chunks)
  sample.py       greedy + temp/top-k/top-p/min-p (per-sampler numpy RNG)
  generate.py     chat template -> prefill -> synchronous decode loop -> streamed text
  weights.py      safetensors load + dotted-key bind onto the model tree
  models/         per-architecture files + registry (qwen3, llama)
  kernels/        fused int8 GEMV/GEMM + size-adaptive dispatcher (M1)
bench/            byte+FLOP roofline; decode tok/s & achieved-BW; crossover bench (M1/M2)
tests/            pure-numpy unit tests + an HF-fp32 parity gate
```

## Setup ([uv](https://docs.astral.sh/uv/))

```bash
cd quartz
uv venv
uv pip install -e ".[reference,dev]"     # numpy/numba/... + torch (HF fp32 oracle)
```

## Run

```bash
uv run python -m quartz.generate --model Qwen/Qwen3-0.6B --prompt "Explain RoPE in one sentence."
```

## Test

```bash
uv run pytest -m "not oracle"                                       # pure numpy, no checkpoint
QUARTZ_PARITY_MODEL=Qwen/Qwen3-0.6B uv run pytest tests/test_parity.py -m oracle
```

## License

[MIT](LICENSE) (code). Model weights carry their own licenses — never committed here.
