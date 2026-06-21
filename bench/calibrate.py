"""Per-class in-situ int8 crossover — the honest single-pass calibration.

bench/kernel_bench.py times each matmul in isolation (cache-favourable). This
instead flips ONE projection class to int8 across all layers, measures the real
end-to-end decode tok/s, and compares to the fp32 baseline — so the weight is read
once per token in a full streaming step, with no cache-reuse inflation. The result
is the data-driven dispatch policy: which classes actually pay off as int8.
"""

from __future__ import annotations

import argparse

from quartz.weights import load_model, resolve_model_path
from quartz.generate import load_tokenizer, _encode_prompt
from quartz.config import GenConfig, QuantConfig
from quartz.kernels.platform import configure_threads
from bench.decode import time_decode

# tied Qwen3-0.6B exposes its head as `embed_tokens`; untied models use `lm_head`.
CLASSES = ["embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser(description="per-class in-situ int8 decode crossover")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()

    configure_threads()
    path = resolve_model_path(args.model)
    tok = load_tokenizer(path)
    prompt_ids = _encode_prompt(tok, "Tell me about large language models.", GenConfig())

    base_model, _ = load_model(path)
    base = time_decode(base_model, prompt_ids, args.tokens)
    print(f"fp32 baseline: {base:.1f} tok/s\n")
    print(f"{'int8 class':<16}{'tok/s':>9}{'vs fp32':>10}{'verdict':>10}")
    for cls in CLASSES:
        model, _ = load_model(path, quant=QuantConfig(default="fp32", include=(cls,)))
        # skip classes that matched no module (e.g. lm_head on a tied model)
        from quartz.quantize import QuantizedLinear, QuantizedEmbedding
        from quartz.weights import _walk_leaves
        if not any(isinstance(m, (QuantizedLinear, QuantizedEmbedding))
                   for _n, m in _walk_leaves(model)):
            continue
        tps = time_decode(model, prompt_ids, args.tokens)
        ratio = tps / base
        verdict = "int8 WIN" if ratio > 1.02 else ("fp32 win" if ratio < 0.98 else "neutral")
        print(f"{cls:<16}{tps:>9.1f}{ratio:>9.2f}x{verdict:>10}")


if __name__ == "__main__":
    main()
