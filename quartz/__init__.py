"""quartz — a transparent, portable CPU LLM inference engine (numpy + Numba).

The CPU sibling of `silica` (which targets Apple Silicon / MLX): same transparent,
single-stream design, but the bottleneck and the levers move. On the CPU, compute
is fp32-BLAS-bound and the bandwidth lever is a *fused* int8 kernel (read int8,
dequant-in-register, fp32 FMA) used only where a matmul is memory-bound.

M0 ships the pure-numpy fp32 runtime + an independent HF-fp32 parity gate; the
fused int8 kernels and the size-adaptive hybrid dispatcher land in M1.
"""

from __future__ import annotations

__version__ = "0.0.0"

from .config import GenConfig, ModelConfig, QuantConfig
from .weights import load_model
from .generate import generate, load_tokenizer

__all__ = [
    "GenConfig",
    "ModelConfig",
    "QuantConfig",
    "load_model",
    "generate",
    "load_tokenizer",
]
