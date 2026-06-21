"""Shared pytest fixtures / markers.

Tier-A tests (kernels + numerics) are pure numpy and run anywhere. The parity
gate (`test_parity.py`) needs torch + transformers + a checkpoint; set
QUARTZ_PARITY_MODEL to a path or repo id (default Qwen/Qwen3-0.6B).
"""

import os

import pytest

PARITY_MODEL = os.environ.get("QUARTZ_PARITY_MODEL", "Qwen/Qwen3-0.6B")


def pytest_configure(config):
    config.addinivalue_line("markers", "oracle: requires torch + transformers + a checkpoint")
    config.addinivalue_line("markers", "slow: longer-running perf/quality sweeps")


@pytest.fixture(scope="session")
def parity_model_id():
    return PARITY_MODEL
