"""Slow: downloads the SAPBERT model on first run (~400 MB) and runs it on CPU.

Skipped unless SQUASHBERT_RUN_SLOW=1. The goal here is just to sanity-check shape
and normalization; correctness of the model is not our concern.
"""

import os

import numpy as np
import pytest

slow = pytest.mark.skipif(
    os.environ.get("SQUASHBERT_RUN_SLOW") != "1",
    reason="set SQUASHBERT_RUN_SLOW=1 to enable",
)


@slow
def test_embed_cls_and_sentence_shapes_and_norms():
    from squashbert.embed import EMBED_DIM, Embedder

    e = Embedder(device="cpu")
    strings = ["metformin", "diabetes", "treats"]
    cls = e.embed_cls(strings, batch_size=2)
    sent = e.embed_sentence(["metformin treats diabetes"], batch_size=1)

    assert cls.shape == (3, EMBED_DIM)
    assert sent.shape == (1, EMBED_DIM)
    assert cls.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(cls, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(sent, axis=1), 1.0, atol=1e-5)


@slow
def test_cls_and_mean_pools_differ_for_longer_input():
    from squashbert.embed import Embedder

    e = Embedder(device="cpu")
    s = "metformin treats type 2 diabetes by activating AMPK"
    cls = e.embed_cls([s], max_length=64)
    mean = e.embed_sentence([s], max_length=64)
    # Same weights, different pooling — cosine should be < 1.
    cos = float((cls * mean).sum(axis=1)[0])
    assert cos < 0.999
