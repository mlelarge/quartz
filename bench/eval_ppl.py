"""Perplexity quality study: fp32 vs int8 configs (the quantization quality cost).

PPL is computed full-context over a small pinned corpus (teacher-forced, fp64
log-softmax for a stable reduction). `--ablate` compares the fp32 baseline against
the hybrid (int8 lm_head), per-group int8, and int8-everywhere recipes — the
quality side of the speed/quality Pareto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quartz.weights import load_model, resolve_model_path
from quartz.generate import load_tokenizer
from quartz.config import QuantConfig

CORPUS = Path(__file__).parent / "data" / "corpus.txt"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def perplexity_from_logits(logits: np.ndarray, target_ids: np.ndarray) -> float:
    """exp(mean NLL): `logits[i]` predicts `target_ids[i]`."""
    lp = log_softmax(logits.astype(np.float64))
    nll = -lp[np.arange(len(target_ids)), np.asarray(target_ids)]
    return float(np.exp(nll.mean()))


def corpus_ppl(model, ids) -> float:
    ids = np.asarray(ids)
    logits = model(ids[None])[0]                      # (L, V)
    return perplexity_from_logits(logits[:-1], ids[1:])


ABLATE = {
    "fp32": None,
    "hybrid (int8 lm_head)": QuantConfig(),
    "hybrid g64": QuantConfig(group_size=64),
    "int8 everywhere": QuantConfig(default="int8"),
}


def main():
    ap = argparse.ArgumentParser(description="perplexity: fp32 vs int8 quantization recipes")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    path = resolve_model_path(args.model)
    tok = load_tokenizer(path)
    text = Path(args.corpus).read_text()
    ids = tok.encode(text)
    print(f"corpus: {len(ids)} tokens")

    configs = ABLATE if args.ablate else {"fp32": None}
    base = None
    print(f"{'config':<26}{'PPL':>9}{'Δ vs fp32':>12}")
    for label, quant in configs.items():
        model, _ = load_model(path, quant=quant)
        ppl = corpus_ppl(model, ids)
        if base is None:
            base = ppl
        delta = f"{100 * (ppl / base - 1):+.1f}%" if base else ""
        print(f"{label:<26}{ppl:>9.3f}{delta:>12}")


if __name__ == "__main__":
    main()
