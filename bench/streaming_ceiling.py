"""Measured CPU streaming-read bandwidth — the honest "% of peak" denominator.

A single-threaded `np.sum` UNDER-measures the achievable bandwidth (one core can't
saturate the memory controller); a multi-threaded parallel read does. Reporting
"% of peak" against the single-sum number would inflate the result several-fold,
and against the chip's *spec* (which only the GPU can reach) would deflate it. So
the bench denominator is this measured *parallel* streaming read.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _parallel_sum(a):
    s = np.float32(0.0)
    for i in prange(a.shape[0]):       # numba detects the reduction; partitions across threads
        s += a[i]
    return s


def _median_gbps(fn, nbytes, runs):
    rates = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        rates.append(nbytes / (time.perf_counter() - t0) / 1e9)
    rates.sort()
    return rates[len(rates) // 2]


def measure(mb: int = 512, runs: int = 9):
    """Return (parallel_gbps, single_sum_gbps) reading an `mb`-MB fp32 array (>> LLC)."""
    a = np.ones(mb * 1024 * 1024 // 4, dtype=np.float32)
    _parallel_sum(a)                   # JIT warm
    a.sum()
    par = _median_gbps(lambda: _parallel_sum(a), a.nbytes, runs)
    single = _median_gbps(a.sum, a.nbytes, runs)
    return par, single


def main():
    ap = argparse.ArgumentParser(description="measured CPU streaming-read bandwidth ceiling")
    ap.add_argument("--mb", type=int, default=512, help="array size in MB (must exceed LLC)")
    ap.add_argument("--runs", type=int, default=9)
    args = ap.parse_args()
    par, single = measure(args.mb, args.runs)
    print(f"array: {args.mb} MB fp32 (>> LLC)")
    print(f"single-thread np.sum : {single:6.1f} GB/s   (under-measures; one core can't saturate)")
    print(f"parallel numba read  : {par:6.1f} GB/s   <- the honest %-of-peak denominator")
    print(f"parallel / single    : {par / single:4.1f}x")


if __name__ == "__main__":
    main()
