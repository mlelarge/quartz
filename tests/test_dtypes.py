"""bf16 -> fp32 upcast is lossless (the safetensors-numpy gotcha)."""

import numpy as np

from quartz.dtypes import upcast_bf16


def test_bf16_known_values():
    # bf16 bit patterns and their exact fp32 values
    cases = {0x3F80: 1.0, 0xC020: -2.5, 0x4049: 3.140625, 0x0000: 0.0, 0xBF80: -1.0}
    u16 = np.array(list(cases.keys()), dtype=np.uint16)
    out = upcast_bf16(u16)
    assert out.dtype == np.float32
    assert out.tolist() == list(cases.values())


def test_bf16_is_high_16_bits_of_fp32():
    rng = np.random.default_rng(0)
    f32 = rng.standard_normal(1000).astype(np.float32)
    # truncate to bf16 (high 16 bits) then upcast -> should equal the truncation
    bits = f32.view(np.uint32)
    bf16 = (bits >> 16).astype(np.uint16)
    back = upcast_bf16(bf16)
    expected = (bf16.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(back, expected)
