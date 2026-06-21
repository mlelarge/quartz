"""KV cache: preallocation in step chunks, offset bookkeeping, valid slices."""

import numpy as np

from quartz.cache import NpKVCache, make_cache


def _kv(b, n_kv, n_new, hd, val):
    return (np.full((b, n_kv, n_new, hd), val, np.float32),
            np.full((b, n_kv, n_new, hd), val, np.float32))


def test_growth_and_offset():
    c = NpKVCache(step=4)
    k, v = _kv(1, 2, 3, 8, 1.0)
    ok, ov = c.update_and_fetch(k, v)
    assert c.offset == 3
    assert ok.shape == (1, 2, 3, 8)
    assert c.keys.shape[2] == 4                      # padded to one step

    k2, v2 = _kv(1, 2, 2, 8, 2.0)                    # crosses the step boundary
    ok2, _ = c.update_and_fetch(k2, v2)
    assert c.offset == 5
    assert ok2.shape == (1, 2, 5, 8)
    assert c.keys.shape[2] == 8                      # grown to two steps
    # earlier tokens preserved, new tokens appended
    assert (ok2[:, :, :3, :] == 1.0).all()
    assert (ok2[:, :, 3:, :] == 2.0).all()


def test_make_cache():
    caches = make_cache(5, step=16)
    assert len(caches) == 5
    assert all(isinstance(c, NpKVCache) and c.offset == 0 for c in caches)
