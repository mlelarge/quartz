"""Platform detection: performance-core count + cpu_info shape."""

from quartz.kernels.platform import performance_cores, cpu_info, select_kernel
from quartz.config import QuantConfig


def test_performance_cores_positive_and_bounded():
    pc = performance_cores()
    assert 1 <= pc <= (cpu_info()["cores"] or pc)      # P-cores ≤ total logical cores


def test_cpu_info_keys():
    info = cpu_info()
    for k in ("arch", "cores", "perf_cores", "has_avx512_vnni", "blas"):
        assert k in info


def test_select_kernel_default_w8a32():
    assert select_kernel(QuantConfig()) == "w8a32"          # M1/M4 default
    assert select_kernel(QuantConfig(kernel="w8a8")) == "w8a8"
