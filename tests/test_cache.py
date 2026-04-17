import numpy as np

from squashbert.cache import EmbeddingCache, edge_type_key
from squashbert.render import EdgeSpec


def test_create_fill_load_roundtrip(tmp_path):
    keys = ["A:1", "A:2", "A:3"]
    cache = EmbeddingCache.create(tmp_path / "cache", keys, dim=4)
    cache.vectors[0] = np.array([1, 0, 0, 0], dtype=np.float16)
    cache.vectors[1] = np.array([0, 1, 0, 0], dtype=np.float16)
    cache.vectors[2] = np.array([0, 0, 1, 0], dtype=np.float16)
    cache.flush()
    del cache

    loaded = EmbeddingCache.load(tmp_path / "cache")
    assert len(loaded) == 3
    assert "A:2" in loaded
    assert not np.isnan(loaded["A:1"]).any()
    np.testing.assert_array_equal(loaded["A:2"], np.array([0, 1, 0, 0], dtype=np.float16))


def test_edge_type_key_distinguishes_direction_and_qualifiers():
    plain = EdgeSpec("biolink:treats")
    qual = EdgeSpec("biolink:causes", "expression", "decreased")

    assert edge_type_key(
        "biolink:SmallMolecule", plain, "biolink:Disease", reverse=False
    ) != edge_type_key(
        "biolink:SmallMolecule", plain, "biolink:Disease", reverse=True
    )
    assert edge_type_key(
        "biolink:SmallMolecule", plain, "biolink:Disease", False
    ) != edge_type_key(
        "biolink:SmallMolecule", qual, "biolink:Disease", False
    )
