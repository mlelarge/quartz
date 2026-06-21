"""Kernel layer: fused int8 GEMV/GEMM (the quartz IP) + platform selection.

M1 ships the validated `w8a32` kernel (int8 weight, fp32 activation) on all
platforms. The int8-activation + AVX-512-VNNI variant (`w8a8`) is deferred to M3
(it needs a real x86 box to validate the VNNI win).
"""

from __future__ import annotations

from .int8 import w8a32_gemv, w8a32_gemm, matmul_int8, warmup

__all__ = ["w8a32_gemv", "w8a32_gemm", "matmul_int8", "warmup"]
