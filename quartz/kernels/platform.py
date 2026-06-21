"""CPU / BLAS detection + thread configuration.

The hybrid decode step calls BOTH numpy/BLAS (fp32 body) and numba prange (int8
lm_head) — sequentially, so they don't overlap, but their thread pools must not
oversubscribe. `configure_threads` pins both to the physical core count.
"""

from __future__ import annotations

import os
import platform
import subprocess

import numpy as np


def performance_cores() -> int:
    """Best default thread count: the number of PERFORMANCE cores.

    On Apple Silicon (and Intel hybrid chips) the efficiency cores are far slower,
    and a numba `prange` that spreads across them barrier-stalls on the slowest
    chunk — a *catastrophic* collapse for the many-small-kernel int8 path (measured:
    16 threads dropped int8-everywhere decode from ~38 to ~24 tok/s on an M3 Max,
    which has 12 P + 4 E cores). Detect P-cores via sysctl on macOS; elsewhere fall
    back to the full logical count (homogeneous cores, or detection unavailable).
    """
    n = os.cpu_count() or 1
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                 capture_output=True, text=True, timeout=2)
            p = int(out.stdout.strip())
            if p > 0:
                return p
        except Exception:
            pass
    return n


def cpu_info() -> dict:
    """arch / cores / AVX-512-VNNI (x86) / BLAS vendor — informs kernel selection."""
    info = {
        "arch": platform.machine().lower(),
        "cores": os.cpu_count(),
        "perf_cores": performance_cores(),
        "has_avx512_vnni": False,
        "blas": "unknown",
    }
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        info["has_avx512_vnni"] = ("avx512_vnni" in flags) or ("avx512vnni" in flags)
    except OSError:
        pass
    try:
        cfg = np.show_config(mode="dicts")
        info["blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    except Exception:
        pass
    return info


def select_kernel(qcfg) -> str:
    """M1: w8a32 everywhere (validated). w8a8 (int8 act + VNNI) is an M3 experiment."""
    return "w8a8" if getattr(qcfg, "kernel", "auto") == "w8a8" else "w8a32"


def configure_threads(n: int | None = None):
    """Pin numba + BLAS thread pools to `n` (default: PERFORMANCE cores, to avoid
    the E-core `prange` collapse). Returns a threadpoolctl context if available
    (use as a `with`), else None."""
    n = n or performance_cores()
    try:
        import numba
        numba.set_num_threads(n)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except Exception:
        return None
