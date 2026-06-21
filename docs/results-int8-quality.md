# quartz int8 quality — 8-bit is essentially free (this model/corpus); the decision is speed

Perplexity of each int8 recipe vs the fp32 baseline, full-context teacher-forced over
a pinned 415-token corpus (`bench/data/corpus.txt`), fp64 log-softmax reduction.
M3 Max, Qwen3-0.6B. `bench/eval_ppl.py --ablate`.

| config | PPL | Δ vs fp32 |
|---|---|---|
| fp32 | 47.328 | +0.0% |
| **hybrid (int8 lm_head, per-row)** | 47.373 | **+0.1%** |
| hybrid (int8 lm_head, group=64) | 47.287 | **−0.1%** |
| int8 everywhere | 47.489 | +0.3% |

(Absolute PPL is high — a 0.6B model, full-context, generic corpus — but the **deltas**
are what matter and they are tiny. The corpus is small (415 tokens), so the single
stronger signal is not this PPL alone but its agreement with the independent M1
generation-parity check below — two unrelated measurements both say "lossless-ish".)

## Reading it

- **8-bit symmetric quantization is quality-free here.** The hybrid recipe (the one
  that ships) costs **+0.1% PPL**. Even quantizing *everything* costs only **+0.3%**.
- This is the decisive complement to the speed results (`docs/results-cpu.md`):
  **"int8 everywhere" is rejected purely on SPEED** (it drops decode below fp32 because
  numba can't beat BLAS on the small cache-resident body matmuls), **not on quality** —
  its perplexity is within a third of a percent of fp32. The dispatch decision is a pure
  speed question, which is why the calibration (`bench/calibrate.py`) is the authority.
- **Per-group (group=64) is a near-free quality refinement** — it slightly *beats* fp32
  PPL here (−0.1%, within noise; finer scales can act as mild regularization) and improves
  worst-case last-logit cosine (0.99998 vs 0.99997 per-row). It is available via
  `QuantConfig(group_size=64)`; per-row is the default for simplicity.
- **Generation is unaffected** (verification, M1): hybrid greedy output is token-for-token
  identical to fp32 on 7/8 prompts (the one divergence is a benign synonym, stays
  coherent), teacher-forced argmax agreement 8/8, worst last-logit cosine 0.99997 — even
  though the tied embedding being int8 means the *input* embeddings are int8-dequantized
  too. The int8 input-embedding drift (worst row cosine 0.99982 over the full vocab) does
  not compound through the 28-layer stack.

## Why 8-bit and not 4-bit (yet)

quartz's thesis is **bandwidth**, and 8-bit already cuts weight traffic 4× vs fp32 while
staying quality-free. 4-bit would halve the lm_head bytes again (a bigger bandwidth lever)
but needs in-register nibble unpacking (a second kernel family) and a real quality study
vs this lossless 8-bit baseline — deferred. The `QuantizedWeight` layout leaves room for a
`bits` field.
