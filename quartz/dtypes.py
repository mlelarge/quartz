"""Safetensors loading for a numpy engine + the bf16 gotcha.

`safetensors.numpy` has NO bfloat16 entry and raises on bf16 tensors — and
Qwen3-0.6B publishes its weights in bf16. So we parse the safetensors container
directly (8-byte header length + JSON header + raw data section) and upcast bf16
to fp32 ourselves: a bf16 value is exactly the top 16 bits of the fp32 with the
same sign/exponent/mantissa-high, so widening is a lossless left-shift:

    fp32_bits = uint16_bits << 16        # zero-pads the low mantissa

This is verified bit-exact (0x3F80 -> 1.0, 0xC020 -> -2.5, 0x4049 -> 3.140625).
"""

from __future__ import annotations

import glob
import json
import struct
from pathlib import Path

import numpy as np

# safetensors dtype string -> numpy dtype (bf16 handled specially below).
_ST_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}


def upcast_bf16(u16: np.ndarray) -> np.ndarray:
    """Lossless bfloat16 (as raw uint16) -> float32."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _load_one(path: Path) -> dict[str, np.ndarray]:
    """Parse a single .safetensors file into {name: ndarray} (bf16 -> fp32)."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        data = fh.read()                       # the rest is the tensor data section
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        raw = data[start:end]
        if dtype == "BF16":
            arr = upcast_bf16(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
        else:
            np_dt = _ST_DTYPES.get(dtype)
            if np_dt is None:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name}")
            arr = np.frombuffer(raw, dtype=np_dt).reshape(shape)
        out[name] = np.ascontiguousarray(arr)
    return out


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load a checkpoint dir: single `model.safetensors` OR sharded (index.json)."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            shards = sorted({v for v in json.load(f)["weight_map"].values()})
        files = [path / s for s in shards]
    else:
        files = [Path(p) for p in glob.glob(str(path / "*.safetensors"))]
    if not files:
        raise FileNotFoundError(f"no .safetensors found under {path}")
    weights: dict[str, np.ndarray] = {}
    for f in files:
        weights.update(_load_one(f))
    return weights
