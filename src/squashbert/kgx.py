"""Stream KGX jsonl files; build a node index and enumerate distinct edge types.

A KGX bundle contains:

- `nodes.jsonl`: one JSON object per line with at least `id`, `name`, `category` (list).
- `edges.jsonl`: one JSON object per line with `subject`, `predicate`, `object`, and
  optional qualifier fields.

For path sampling we need:

1. Each node's *most specific* biolink category (not list order — actual hierarchy depth).
2. A category -> [node_id] index so we can pick a random node of a given category.
3. The set of distinct "edge types" — (subject_category, EdgeSpec, object_category)
   triples — so we can sample edges type-compatibly with a preceding node.

The edge-type set is built by one streaming pass over edges.jsonl with the node index
held in memory (~400 MB for 2M nodes).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import orjson

from .biolink import most_specific_category
from .render import EdgeSpec, UnsupportedQualifierPattern, edge_spec_from_kgx


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    name: str
    category: str  # most-specific biolink CURIE


@dataclass(frozen=True, slots=True)
class EdgeType:
    subject_category: str
    spec: EdgeSpec
    object_category: str


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per line of a jsonl file. Fast path via orjson."""
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def load_nodes(path: str | Path) -> dict[str, Node]:
    """Load all nodes into an id -> Node dict.

    Skips nodes with no `name` (can't embed them) and nodes whose categories contain
    no known biolink category (can't sample them by type).
    """
    nodes: dict[str, Node] = {}
    for rec in iter_jsonl(path):
        name = rec.get("name")
        if not name:
            continue
        try:
            cat = most_specific_category(rec.get("category") or [])
        except ValueError:
            continue
        nodes[rec["id"]] = Node(id=rec["id"], name=name, category=cat)
    return nodes


def build_category_index(nodes: dict[str, Node]) -> dict[str, list[str]]:
    """category CURIE -> list of node ids with that most-specific category."""
    idx: dict[str, list[str]] = {}
    for nid, n in nodes.items():
        idx.setdefault(n.category, []).append(nid)
    return idx


def collect_edge_types(
    edges_path: str | Path, nodes: dict[str, Node]
) -> tuple[list[EdgeType], dict[str, int]]:
    """Scan edges.jsonl once and return (unique_edge_types, skip_counts).

    `skip_counts` reports why edges were skipped: missing endpoint nodes,
    unsupported qualifier patterns, etc. — useful for sanity-checking a new KGX.
    """
    seen: set[EdgeType] = set()
    skips = {"missing_endpoint": 0, "unsupported_qualifier": 0}
    for rec in iter_jsonl(edges_path):
        s_id = rec.get("subject")
        o_id = rec.get("object")
        s_node = nodes.get(s_id) if s_id else None
        o_node = nodes.get(o_id) if o_id else None
        if s_node is None or o_node is None:
            skips["missing_endpoint"] += 1
            continue
        try:
            spec = edge_spec_from_kgx(rec)
        except UnsupportedQualifierPattern:
            skips["unsupported_qualifier"] += 1
            continue
        seen.add(
            EdgeType(
                subject_category=s_node.category,
                spec=spec,
                object_category=o_node.category,
            )
        )
    return list(seen), skips
