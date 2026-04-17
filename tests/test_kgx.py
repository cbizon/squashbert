import orjson

from squashbert.kgx import (
    EdgeType,
    build_category_index,
    collect_edge_types,
    load_nodes,
)
from squashbert.render import EdgeSpec


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def test_load_nodes_picks_most_specific_category(tmp_path):
    p = tmp_path / "nodes.jsonl"
    _write_jsonl(
        p,
        [
            {
                "id": "A:1",
                "name": "metformin",
                "category": ["biolink:NamedThing", "biolink:SmallMolecule"],
            },
            {
                "id": "D:1",
                "name": "diabetes",
                "category": ["biolink:Disease", "biolink:NamedThing"],
            },
        ],
    )
    nodes = load_nodes(p)
    assert nodes["A:1"].category == "biolink:SmallMolecule"
    assert nodes["D:1"].category == "biolink:Disease"


def test_load_nodes_skips_missing_name_or_category(tmp_path):
    p = tmp_path / "nodes.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "X", "category": ["biolink:Gene"]},  # no name
            {"id": "Y", "name": "foo", "category": []},  # no category
            {"id": "Z", "name": "bar", "category": ["not:a_category"]},  # unknown
            {"id": "G:1", "name": "gene1", "category": ["biolink:Gene"]},
        ],
    )
    nodes = load_nodes(p)
    assert set(nodes) == {"G:1"}


def test_build_category_index(tmp_path):
    p = tmp_path / "nodes.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "G:1", "name": "gene1", "category": ["biolink:Gene"]},
            {"id": "G:2", "name": "gene2", "category": ["biolink:Gene"]},
            {"id": "D:1", "name": "dz", "category": ["biolink:Disease"]},
        ],
    )
    nodes = load_nodes(p)
    idx = build_category_index(nodes)
    assert set(idx["biolink:Gene"]) == {"G:1", "G:2"}
    assert idx["biolink:Disease"] == ["D:1"]


def test_collect_edge_types_dedupes_and_skips(tmp_path):
    npath = tmp_path / "nodes.jsonl"
    epath = tmp_path / "edges.jsonl"
    _write_jsonl(
        npath,
        [
            {"id": "A:1", "name": "metformin", "category": ["biolink:SmallMolecule"]},
            {"id": "D:1", "name": "diabetes", "category": ["biolink:Disease"]},
            {"id": "D:2", "name": "t2dm", "category": ["biolink:Disease"]},
            {"id": "G:1", "name": "glp1r", "category": ["biolink:Gene"]},
        ],
    )
    _write_jsonl(
        epath,
        [
            # Two edges sharing the same (subj_cat, spec, obj_cat) → one type.
            {"subject": "A:1", "predicate": "biolink:treats", "object": "D:1"},
            {"subject": "A:1", "predicate": "biolink:treats", "object": "D:2"},
            # Qualified edge — separate type.
            {
                "subject": "A:1",
                "predicate": "biolink:affects",
                "qualified_predicate": "biolink:causes",
                "object_aspect_qualifier": "expression",
                "object_direction_qualifier": "decreased",
                "object": "G:1",
            },
            # Missing endpoint — skipped.
            {"subject": "A:1", "predicate": "biolink:treats", "object": "MISSING:1"},
            # Unsupported qualifier — skipped.
            {
                "subject": "A:1",
                "predicate": "biolink:affects",
                "qualified_predicate": "biolink:causes",
                "subject_aspect_qualifier": "hydrolysis",
                "subject_direction_qualifier": "increased",
                "object": "G:1",
            },
        ],
    )
    nodes = load_nodes(npath)
    types, skips = collect_edge_types(epath, nodes)
    assert len(types) == 2
    assert skips == {"missing_endpoint": 1, "unsupported_qualifier": 1}
    assert EdgeType(
        "biolink:SmallMolecule", EdgeSpec("biolink:treats"), "biolink:Disease"
    ) in types
    assert EdgeType(
        "biolink:SmallMolecule",
        EdgeSpec("biolink:causes", "expression", "decreased"),
        "biolink:Gene",
    ) in types
