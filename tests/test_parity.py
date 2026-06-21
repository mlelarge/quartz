"""M0 parity gate — the most important test in the project.

quartz has no same-backend reference (silica leaned on mlx-lm), so the
independent oracle is **HuggingFace transformers fp32 on CPU** — a different
implementation. We load quartz in fp32 too, so the only differences are
kernel/accumulation order, making teacher-forced per-position argmax agreement a
tight, meaningful check. Skips unless torch + transformers + a checkpoint exist
(`uv pip install -e ".[reference]"`).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from quartz.weights import load_model, resolve_model_path
from quartz.generate import load_tokenizer, generate
from quartz.config import GenConfig

PROMPTS = [
    "The capital of France is",
    "Café crème costs €3.50 — déjà vu? 🤔",   # multibyte / emoji (detok stress)
]


@pytest.fixture(scope="module")
def oracle(parity_model_id):
    from transformers import AutoModelForCausalLM

    try:
        path = resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")

    q_model, _ = load_model(path)               # fp32
    hf = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
    hf.eval()
    tok = load_tokenizer(path)
    return q_model, hf, tok


@pytest.mark.oracle
@pytest.mark.parametrize("prompt", PROMPTS)
def test_teacher_forced_argmax_matches_hf_fp32(oracle, prompt):
    """quartz-fp32 argmax agrees with HF-fp32 at every position of the same input."""
    q_model, hf, tok = oracle
    ids = tok.encode(prompt)

    q_logits = q_model(np.array([ids]))[0]               # (L, V)
    with torch.no_grad():
        h_logits = hf(torch.tensor([ids])).logits[0]     # (L, V)

    q_arg = np.argmax(q_logits, axis=-1).tolist()
    h_arg = h_logits.argmax(dim=-1).tolist()

    assert q_arg[-1] == h_arg[-1], f"next-token argmax differs: q={q_arg[-1]} hf={h_arg[-1]}"
    rate = sum(a == b for a, b in zip(q_arg, h_arg)) / len(q_arg)
    assert rate >= 0.9, f"only {rate:.0%} of positions agree with HF fp32"


@pytest.mark.oracle
def test_last_logits_cosine_with_hf_fp32(oracle):
    """Numerical agreement (not just argmax): next-token logit cosine ~1.0."""
    q_model, hf, tok = oracle
    ids = tok.encode(PROMPTS[0])

    q_logits = q_model(np.array([ids]))[0, -1]
    with torch.no_grad():
        h_logits = hf(torch.tensor([ids])).logits[0, -1]

    s = q_logits.astype(np.float64)
    h = h_logits.detach().to(torch.float64).numpy()
    cos = float(np.dot(s, h) / (np.linalg.norm(s) * np.linalg.norm(h)))
    assert cos > 0.999, f"logit cosine vs HF fp32 too low: {cos:.5f}"


@pytest.mark.oracle
def test_string_decode_handles_multibyte(oracle):
    """Detokenizer must not corrupt multibyte characters."""
    q_model, _, tok = oracle
    cfg = GenConfig(max_tokens=16, temperature=0.0)
    out = generate(q_model, tok, PROMPTS[1], cfg, stream=False)
    assert "�" not in out
