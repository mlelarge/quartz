"""End-to-end decode bench: fp32 vs hybrid (int8 lm_head) vs int8-everywhere.

This is the HONEST in-situ measurement (a real streaming decode, each weight read
once per token — no cache-reuse inflation). Reports tok/s and achieved bandwidth
(byte_budget x tok/s) per config, so the hybrid's win and the "int8 everywhere
loses" anti-result both show.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import numpy as np

from quartz.weights import load_model, resolve_model_path
from quartz.generate import load_tokenizer, _encode_prompt
from quartz.config import GenConfig, QuantConfig
from quartz.cache import make_cache
from quartz.sample import make_sampler
from quartz.kernels.platform import configure_threads, cpu_info
from bench.roofline import byte_budget


def time_decode(model, prompt_ids, n_tokens, *, warmup=3, runs=5):
    samp = make_sampler(GenConfig())

    def run():
        cache = make_cache(len(model.layers))
        y = samp(model.decode_logits(np.array([prompt_ids]), cache=cache)[:, -1, :])
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            y = samp(model.decode_logits(np.array([[int(y[0])]]), cache=cache)[:, -1, :])
        return n_tokens / (time.perf_counter() - t0)

    for _ in range(warmup):
        run()
    rates = sorted(run() for _ in range(runs))
    return rates[len(rates) // 2]


CONFIGS = {
    "fp32": (None, dict()),
    "hybrid (int8 lm_head)": (QuantConfig(), dict(int8_lm_head=True)),
    "int8 everywhere": (QuantConfig(default="int8"), dict(int8=True, int8_lm_head=True)),
}


def main():
    ap = argparse.ArgumentParser(description="end-to-end decode tok/s: fp32 vs int8 configs")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--context-len", type=int, default=0, help="for the achieved-bandwidth byte model")
    ap.add_argument("--bandwidth-gbps", type=float, default=None, help="measured CPU streaming ceiling")
    args = ap.parse_args()

    # Hold the BLAS thread-limit context for the whole bench (on x86/OpenBLAS,
    # dropping it would release the cap immediately and let pools oversubscribe).
    with (configure_threads() or nullcontext()):
        path = resolve_model_path(args.model)
        tok = load_tokenizer(path)
        prompt_ids = _encode_prompt(tok, "Tell me about large language models in detail.", GenConfig())
        info = cpu_info()
        print(f"machine: {info['arch']} {info['cores']}c, BLAS={info['blas']}, "
              f"VNNI={info['has_avx512_vnni']}")

        # tok/s + ms/tok are the honest speed FoM (always shown). The byte-model
        # bandwidth is shown ONLY with --bandwidth-gbps, because byte-model x tok/s
        # is NOT a speed proxy: int8 cuts the byte denominator, so a faster int8
        # config can show a *lower* achieved GB/s, and int8-everywhere's low figure
        # reflects fewer bytes moved, not bandwidth headroom (its real bottleneck is
        # numba kernel compute on the small body matmuls).
        show_bw = args.bandwidth_gbps is not None
        header = f"{'config':<26}{'tok/s':>9}{'ms/tok':>9}"
        if show_bw:
            header += f"{'byte-GB/s':>11}{'% peak':>9}"
        print(header)

        for label, (quant, byte_kw) in CONFIGS.items():
            model, cfg = load_model(path, quant=quant)
            tps = time_decode(model, prompt_ids, args.tokens)
            row = f"{label:<26}{tps:>9.1f}{1e3 / tps:>9.2f}"
            if show_bw:
                gbps = byte_budget(cfg, args.context_len, **byte_kw).achieved_bandwidth_gbps(tps)
                row += f"{gbps:>11.1f}{100 * gbps / args.bandwidth_gbps:>8.0f}%"
            print(row)
        if show_bw:
            print("note: byte-GB/s = byte-model x tok/s (assumes bandwidth-bound); for "
                  "int8-everywhere the bottleneck is kernel compute, not bandwidth.")


if __name__ == "__main__":
    main()
