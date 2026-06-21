"""Attention block locks: causal mask, GQA repeat (not tile), SDPA vs reference."""

import numpy as np

from quartz.attention import causal_additive_mask, sdpa
from quartz.ops import softmax


def test_causal_mask_none_for_single_token():
    assert causal_additive_mask(1, 0) is None
    assert causal_additive_mask(1, 17) is None       # decode: attend all cached keys


def test_causal_mask_shape_and_values():
    m = causal_additive_mask(3, 2)                    # query positions 2,3,4 over keys 0..4
    assert m.shape == (3, 5)
    # query 0 (abs pos 2) attends keys 0,1,2 ; not 3,4
    assert (m[0, :3] == 0).all()
    assert np.isneginf(m[0, 3:]).all()
    # last query (abs pos 4) attends everything
    assert (m[2] == 0).all()


def test_gqa_repeat_is_contiguous_not_tile():
    # n_kv=2, n_rep=3 -> head order must be [0,0,0,1,1,1], NOT [0,1,0,1,0,1]
    k = np.arange(2).reshape(1, 2, 1, 1).astype(np.float32)
    rep = np.repeat(k, 3, axis=1)[0, :, 0, 0]
    assert rep.tolist() == [0, 0, 0, 1, 1, 1]
    assert np.tile(k, (1, 3, 1, 1))[0, :, 0, 0].tolist() == [0, 1, 0, 1, 0, 1]


def test_sdpa_matches_reference():
    rng = np.random.default_rng(0)
    B, nkv, nrep, L, D = 1, 2, 2, 4, 8
    nq = nkv * nrep
    q = rng.standard_normal((B, nq, L, D)).astype(np.float32)
    k = rng.standard_normal((B, nkv, L, D)).astype(np.float32)
    v = rng.standard_normal((B, nkv, L, D)).astype(np.float32)
    scale = D ** -0.5
    mask = causal_additive_mask(L, 0)
    out = sdpa(q, k, v, scale=scale, mask=mask, n_rep=nrep)

    # naive reference with explicit head repeat
    kk = np.repeat(k, nrep, axis=1)
    vv = np.repeat(v, nrep, axis=1)
    ref = np.empty_like(out)
    for h in range(nq):
        scores = (q[0, h] @ kk[0, h].T) * scale + mask
        ref[0, h] = softmax(scores, axis=-1) @ vv[0, h]
    assert np.allclose(out, ref, atol=1e-5)


def test_sdpa_decode_no_mask():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((1, 4, 1, 8)).astype(np.float32)   # single query token
    k = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)   # 6 cached keys
    v = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
    out = sdpa(q, k, v, scale=8 ** -0.5, mask=None, n_rep=2)
    assert out.shape == (1, 4, 1, 8)
