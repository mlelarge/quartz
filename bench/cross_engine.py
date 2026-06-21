"""Cross-engine yardstick: quartz (numpy+numba) vs llama.cpp (C++/SIMD), same CPU.

The honest external mirror. Both decode Qwen3-0.6B on the CPU; llama.cpp via
llama-bench with `-ngl 0` (no GPU offload). Each engine is benched at ITS BEST
thread count — both collapse if a parallel loop spreads across the efficiency
cores (numba on E-cores, or llama.cpp `-t 16`), so a naive all-cores run would
slander whichever engine you forgot to tune.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import shutil
import subprocess

from quartz.weights import load_model, resolve_model_path
from quartz.generate import load_tokenizer, _encode_prompt
from quartz.config import GenConfig, QuantConfig
from quartz.kernels.platform import configure_threads, performance_cores, cpu_info
from bench.decode import time_decode


def find_gguf(name: str) -> str | None:
    base = os.path.expanduser("~/.cache/huggingface/hub")
    hits = glob.glob(f"{base}/**/{name}", recursive=True)
    return hits[0] if hits else None


def llama_cpu_tps(binp: str, gguf: str, threads: int, n: int = 96) -> float | None:
    """Decode (tg) tok/s for llama.cpp on CPU at a given thread count, via `-o csv`."""
    cmd = [binp, "-m", gguf, "-ngl", "0", "-t", str(threads), "-p", "0", "-n", str(n),
           "-r", "3", "-o", "csv"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for row in csv.DictReader(io.StringIO(out)):
        try:
            if int(row.get("n_gen", "0")) > 0:      # the tg (decode) row
                return float(row["avg_ts"])
        except (ValueError, KeyError):
            continue
    return None


def llama_best_cpu(binp: str, gguf: str, threads=(4, 8, 12)):
    vals = [(t, llama_cpu_tps(binp, gguf, t)) for t in threads]
    vals = [(t, v) for t, v in vals if v]
    return max(vals, key=lambda x: x[1]) if vals else (None, None)


def main():
    ap = argparse.ArgumentParser(description="quartz vs llama.cpp CPU, same model")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--gguf-q8", default=None)
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    info = cpu_info()
    pc = performance_cores()
    configure_threads()                              # quartz -> P-cores
    print(f"machine: {info['arch']} {info['cores']}c ({pc} P-cores), BLAS={info['blas']}")

    path = resolve_model_path(args.model)
    tok = load_tokenizer(path)
    pids = _encode_prompt(tok, "Tell me about large language models.", GenConfig())
    q_fp32 = time_decode(load_model(path)[0], pids, args.tokens)
    q_hyb = time_decode(load_model(path, quant=QuantConfig())[0], pids, args.tokens)

    print(f"\n{'engine / config':<34}{'tok/s':>8}{'threads':>9}")
    print(f"{'quartz fp32':<34}{q_fp32:>8.1f}{pc:>9}")
    print(f"{'quartz hybrid (int8 lm_head)':<34}{q_hyb:>8.1f}{pc:>9}")

    binp = shutil.which("llama-bench") or "/opt/homebrew/bin/llama-bench"
    if not (shutil.which("llama-bench") or os.path.exists(binp)):
        print("\n(llama-bench not found — install llama.cpp to compare)")
        return

    q8 = args.gguf_q8 or find_gguf("Qwen3-0.6B-Q8_0.gguf")
    q4 = find_gguf("Qwen3-0.6B-Q4_K_M.gguf")
    llama_q8 = None
    if q8:
        t, llama_q8 = llama_best_cpu(binp, q8)
        print(f"{'llama.cpp Q8_0 (CPU, best -t)':<34}{llama_q8:>8.1f}{t:>9}")
    if q4:
        t, v = llama_best_cpu(binp, q4)
        print(f"{'llama.cpp Q4_K_M (CPU, best -t)':<34}{v:>8.1f}{t:>9}  (4-bit, context)")

    if llama_q8:
        print(f"\nquartz hybrid is {q_hyb / llama_q8:.2f}x llama.cpp Q8_0 CPU "
              f"(llama.cpp is {llama_q8 / q_hyb:.1f}x faster) — the transparency tax: "
              f"quartz's fp32 body is bandwidth-competitive but moves more bytes, and the "
              f"per-op overhead floor blocks an efficient all-int8 body (see results-cpu.md).")


if __name__ == "__main__":
    main()
