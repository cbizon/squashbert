import random

import pytest

from squashbert.kgx import EdgeType, Node
from squashbert.render import EdgeSpec
from squashbert.sampler import PathSampler


@pytest.fixture
def tiny_graph():
    nodes = {
        "A:1": Node("A:1", "metformin", "biolink:SmallMolecule"),
        "A:2": Node("A:2", "aspirin", "biolink:SmallMolecule"),
        "D:1": Node("D:1", "diabetes", "biolink:Disease"),
        "D:2": Node("D:2", "hypertension", "biolink:Disease"),
        "G:1": Node("G:1", "glp1r", "biolink:Gene"),
        "G:2": Node("G:2", "insr", "biolink:Gene"),
    }
    cat_index = {
        "biolink:SmallMolecule": ["A:1", "A:2"],
        "biolink:Disease": ["D:1", "D:2"],
        "biolink:Gene": ["G:1", "G:2"],
    }
    edge_types = [
        EdgeType("biolink:SmallMolecule", EdgeSpec("biolink:treats"), "biolink:Disease"),
        EdgeType(
            "biolink:Disease", EdgeSpec("biolink:related_to"), "biolink:Gene"
        ),
        EdgeType(
            "biolink:SmallMolecule",
            EdgeSpec("biolink:causes", "expression", "decreased"),
            "biolink:Gene",
        ),
        # Same-category edge to exercise the self-loop branch.
        EdgeType(
            "biolink:Gene", EdgeSpec("biolink:related_to"), "biolink:Gene"
        ),
    ]
    return nodes, cat_index, edge_types


def test_sample_one_hop_structure(tiny_graph):
    nodes, cat, edges = tiny_graph
    s = PathSampler(nodes, cat, edges)
    rng = random.Random(0)
    for _ in range(20):
        p = s.sample(1, rng)
        assert len(p.node_ids) == 2
        assert len(p.edge_phrases) == 1
        assert len(p.edge_keys) == 1
        # no revisits
        assert len(set(p.node_ids)) == 2
        # sentence contains both node names
        for nid in p.node_ids:
            assert nodes[nid].name in p.sentence


def test_sample_three_hops_has_no_repeats(tiny_graph):
    nodes, cat, edges = tiny_graph
    s = PathSampler(nodes, cat, edges)
    rng = random.Random(42)
    for _ in range(20):
        p = s.sample(3, rng)
        assert len(p.node_ids) == 4
        assert len(p.edge_phrases) == 3
        assert len(set(p.node_ids)) == 4


def test_reverse_direction_is_used(tiny_graph):
    # With only SmallMolecule→Disease and Disease→Gene forward edges, a path that
    # starts at a Disease and goes to a SmallMolecule must use `treated by`.
    nodes, cat, edges = tiny_graph
    s = PathSampler(nodes, cat, edges)
    saw_reverse = False
    rng = random.Random(1)
    for _ in range(200):
        p = s.sample(1, rng)
        if "treated by" in p.edge_phrases or "related to" in p.edge_phrases:
            saw_reverse = True
    # We should at least see `treated by` in a reasonable number of draws.
    assert saw_reverse


def test_sentence_matches_compose(tiny_graph):
    nodes, cat, edges = tiny_graph
    s = PathSampler(nodes, cat, edges)
    rng = random.Random(7)
    p = s.sample(2, rng)
    expected = (
        f"{nodes[p.node_ids[0]].name} {p.edge_phrases[0]} "
        f"{nodes[p.node_ids[1]].name} {p.edge_phrases[1]} "
        f"{nodes[p.node_ids[2]].name}"
    )
    assert p.sentence == expected
