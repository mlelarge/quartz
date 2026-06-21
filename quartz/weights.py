"""Load a checkpoint into a quartz model.

Mirrors silica's loader (single-file + sharded safetensors, tied-embedding drop)
but in pure numpy: weights are bound by walking HF-style dotted keys onto the
plain-object model tree, whose attribute names match HF exactly. bf16 is upcast
to fp32 at read time (see dtypes.py); M0 is fp32 throughout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .dtypes import load_safetensors
from .models import build_model


def resolve_model_path(model: str | Path) -> Path:
    """Local dir as-is; otherwise download the HF snapshot (weights + config)."""
    p = Path(model)
    if p.exists():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise FileNotFoundError(
            f"{model} not found locally and huggingface_hub is not installed."
        ) from e
    return Path(
        snapshot_download(
            repo_id=str(model),
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"],
        )
    )


def set_param(root, key: str, arr: np.ndarray) -> None:
    """Assign `arr` onto `root` at the dotted HF key (e.g.
    'model.layers.0.self_attn.q_proj.weight'). Numeric components index lists."""
    *path, attr = key.split(".")
    obj = root
    for p in path:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, attr, arr)


def set_module(root, name: str, new_module) -> None:
    """Replace the module at dotted `name` (e.g. 'model.embed_tokens') with
    `new_module` — used by the int8 quantizer to swap a Linear/Embedding leaf."""
    *path, last = name.split(".")
    obj = root
    for p in path:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


def _is_module(x) -> bool:
    return (
        type(x).__module__.startswith("quartz")
        and hasattr(x, "__dict__")
        and not isinstance(x, np.ndarray)
    )


def _walk_leaves(obj, prefix: str = ""):
    """Yield (dotted_name, leaf_module) for every module carrying a `weight`."""
    if hasattr(obj, "weight"):
        yield prefix, obj
        return
    for name, val in vars(obj).items():
        child = f"{prefix}.{name}" if prefix else name
        if isinstance(val, list):
            for i, item in enumerate(val):
                if _is_module(item):
                    yield from _walk_leaves(item, f"{child}.{i}")
        elif _is_module(val):
            yield from _walk_leaves(val, child)


def bind(net, weights: dict[str, np.ndarray]) -> None:
    """Bind a name->array dict onto the model, then verify nothing is unbound."""
    for key, arr in weights.items():
        set_param(net, key, arr)
    # A leaf is unbound if its weight is missing OR a required bias is still the
    # Ellipsis sentinel (Linear(bias=True) whose `.bias` key was absent) — without
    # the bias check an attention_bias=True checkpoint would load with biases
    # silently dropped and emit wrong logits.
    unbound = [
        name for name, m in _walk_leaves(net)
        if getattr(m, "weight", None) is None or getattr(m, "bias", None) is ...
    ]
    if unbound:
        raise ValueError(f"unbound parameters after load: {unbound[:8]}"
                         f"{' ...' if len(unbound) > 8 else ''}")


def load_model(model: str | Path = "Qwen/Qwen3-0.6B", *, dtype=np.float32, quant=None):
    """Build and load a quartz model (architecture chosen by the registry).

    Loads an fp checkpoint and computes in `dtype` (fp32). Pass a `QuantConfig`
    to quantize policy-selected weights to int8 at load (M1 fused-kernel path);
    `quant=None` keeps the model pure fp32 (no numba import).
    """
    path = resolve_model_path(model)
    cfg = ModelConfig.from_json(path / "config.json")
    net = build_model(cfg)

    weights = load_safetensors(path)
    weights = net.sanitize(weights)
    if cfg.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    weights = {k: (v.astype(dtype) if v.dtype.kind == "f" else v) for k, v in weights.items()}

    bind(net, weights)
    if quant is not None:
        from .quantize import quantize_model           # lazy: only this path needs numba
        quantize_model(net, quant)
    return net, cfg
