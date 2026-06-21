"""Where does a decode step's time go? — the per-op-overhead decomposition.

M1 found the fused int8 lm_head is 4.25x in isolation but only 1.21x end-to-end.
The reason: at batch=1 a quartz decode step is a long chain of small numpy ops
(28 layers x q/k/v/o + gate/up/down + 2 RMSNorms + RoPE + SDPA), and the Python
per-call dispatch + small-array materialization is a large share of the wall time
— so shrinking one matmul (the lm_head) only moves its slice.

This reports two things: (1) a cProfile view (call counts expose the dispatch
churn), and (2) a T_step vs T_matmul-isolated split (how much of the step is even
spent in the GEMMs). The isolated matmul time is cache-favourable, so it is a
LOWER bound on matmul share / UPPER bound on overhead — the conclusion is robust.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from quartz.weights import load_model, resolve_model_path, _walk_leaves
from quartz.ops import Linear, Embedding
from quartz.generate import load_tokenizer, _encode_prompt
from quartz.config import GenConfig, QuantConfig
from quartz.cache import make_cache
from quartz.sample import make_sampler


def _run_decode(model, prompt_ids, n):
    samp = make_sampler(GenConfig())
    cache = make_cache(len(model.layers))
    y = samp(model.decode_logits(np.array([prompt_ids]), cache=cache)[:, -1, :])
    for _ in range(n):
        y = samp(model.decode_logits(np.array([[int(y[0])]]), cache=cache)[:, -1, :])


def _median(fn, runs):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def _isolated_matmul_ms(model):
    """Sum of isolated batch=1 matmul times for one decode step (all layers + head)."""
    rng = np.random.default_rng(0)
    total = 0.0
    for _name, m in _walk_leaves(model):
        if not isinstance(m, (Linear, Embedding)):
            continue
        W = np.ascontiguousarray(m.weight, dtype=np.float32)
        N, K = W.shape
        x = rng.standard_normal((1, K)).astype(np.float32)
        total += _median(lambda: x @ W.T, 20)
    return total * 1e3


def main():
    ap = argparse.ArgumentParser(description="decode-step time decomposition (matmul vs overhead)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--quant", choices=["none", "lm_head", "int8"], default="none")
    args = ap.parse_args()

    quant = {"none": None, "lm_head": QuantConfig(),
             "int8": QuantConfig(default="int8")}[args.quant]
    path = resolve_model_path(args.model)
    model, _ = load_model(path, quant=quant)
    tok = load_tokenizer(path)
    prompt_ids = _encode_prompt(tok, "Tell me about large language models.", GenConfig())

    _run_decode(model, prompt_ids, 3)                         # warm

    t_step = _median(lambda: _run_decode(model, prompt_ids, args.tokens), 5) / args.tokens * 1e3
    t_mm = _isolated_matmul_ms(model)
    print(f"quant={args.quant}")
    print(f"  full decode step : {t_step:7.2f} ms/token  ({1e3 / t_step:.1f} tok/s)")
    print(f"  isolated matmuls : {t_mm:7.2f} ms/token  ({100 * t_mm / t_step:.0f}% of the step, "
          f"cache-favourable lower bound)")
    print(f"  -> overhead+rest : {t_step - t_mm:7.2f} ms/token  ({100 * (1 - t_mm / t_step):.0f}% — "
          f"norms/rope/softmax/cache + Python per-op dispatch)")

    pr = cProfile.Profile()
    pr.enable()
    _run_decode(model, prompt_ids, args.tokens)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print("\ncProfile (top by cumulative time; note the call counts = dispatch churn):")
    print("\n".join(s.getvalue().splitlines()[4:18]))


if __name__ == "__main__":
    main()
