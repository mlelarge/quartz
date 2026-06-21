"""Generation: chat template -> prefill -> decode loop -> streamed text.

Unlike silica's MLX loop (which overlaps a host<->device sync with the next
step's GPU compute via `mx.async_eval`), numpy is synchronous and eager: there is
no device queue to hide, so the loop is a plain `while`. That simplification —
and the absence of any async lever — is itself a CPU-vs-GPU data point.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .config import GenConfig, ModelConfig
from .cache import make_cache
from .sample import make_sampler
from .detokenize import IncrementalDetokenizer


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path))


def _encode_prompt(tokenizer, prompt: str, cfg: GenConfig) -> list[int]:
    if cfg.use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)
    return list(tokenizer.encode(prompt))


def generate_step(model, prompt_ids: list[int], cfg: GenConfig, eos_ids: tuple[int, ...]) -> Iterator[int]:
    """Yield generated token ids one at a time (greedy/sampled)."""
    if cfg.max_tokens <= 0:
        return
    if not prompt_ids:
        raise ValueError("empty prompt: nothing to prefill (pass a non-empty "
                         "prompt or enable the chat template)")
    sampler = make_sampler(cfg)
    cache = make_cache(len(model.layers))

    def step(tokens: np.ndarray) -> np.ndarray:
        logits = model.decode_logits(tokens, cache=cache)[:, -1, :]   # last position only
        return sampler(logits)

    y = step(np.array(prompt_ids, dtype=np.int64)[None])   # prefill -> first token
    n = 0
    while True:
        tok = int(y[0])
        yield tok
        n += 1
        if tok in eos_ids or n >= cfg.max_tokens:
            break
        y = step(np.array([[tok]], dtype=np.int64))


def generate(model, tokenizer, prompt: str, cfg: GenConfig | None = None, *, stream: bool = True) -> str:
    """Generate text for `prompt`. Returns the full decoded string."""
    cfg = cfg or GenConfig()
    mcfg: ModelConfig = model.config
    eos_ids = set(mcfg.eos_token_ids)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    prompt_ids = _encode_prompt(tokenizer, prompt, cfg)
    detok = IncrementalDetokenizer(tokenizer, stop=cfg.stop)

    out = []
    for token in generate_step(model, prompt_ids, cfg, tuple(eos_ids)):
        if token in eos_ids:
            break
        segment = detok.add_token(token)
        if segment:
            out.append(segment)
            if stream:
                print(segment, end="", flush=True)
        if detok.finished:
            break
    flush = detok.finalize()
    if flush:
        out.append(flush)
        if stream:
            print(flush, end="", flush=True)
    if stream:
        print()
    return "".join(out)


def main():
    import argparse

    from .weights import load_model, resolve_model_path
    from .config import QuantConfig

    ap = argparse.ArgumentParser(description="quartz greedy generation (numpy fp32 + optional int8)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="Give me a short introduction to large language models.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--quant", choices=["none", "lm_head", "int8"], default="none",
                    help="none=fp32; lm_head=hybrid (int8 lm_head, fp32 body); int8=everything int8")
    args = ap.parse_args()

    quant = {"none": None, "lm_head": QuantConfig(),
             "int8": QuantConfig(default="int8")}[args.quant]
    model, _ = load_model(args.model, quant=quant)
    tokenizer = load_tokenizer(resolve_model_path(args.model))
    cfg = GenConfig(max_tokens=args.max_tokens, temperature=args.temp,
                    use_chat_template=not args.no_chat_template)
    generate(model, tokenizer, args.prompt, cfg)


if __name__ == "__main__":
    main()
