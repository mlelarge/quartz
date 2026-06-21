"""Model registry — maps the HF `architectures` field to a quartz model class.
Each per-architecture module declares an `ARCHITECTURES` tuple naming the classes
it provides; adding a model is: drop a file, list its architecture, done.
"""

from __future__ import annotations

from . import qwen3, llama

REGISTRY: dict[str, type] = {}


def _register(module) -> None:
    for arch in getattr(module, "ARCHITECTURES", ()):
        REGISTRY[arch] = getattr(module, arch)


for _m in (qwen3, llama):
    _register(_m)


def model_class(architectures) -> type:
    for arch in architectures:
        if arch in REGISTRY:
            return REGISTRY[arch]
    raise ValueError(
        f"no quartz model for architectures {list(architectures)}; "
        f"registered: {sorted(REGISTRY)}"
    )


def build_model(cfg):
    """Instantiate the model class for `cfg.architectures`."""
    return model_class(cfg.architectures)(cfg)
