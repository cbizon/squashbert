"""Sample random n-hop paths through the schema of a KG.

We don't use the actual edges in the graph — we sample *type-compatible* paths: at
each hop, pick any edge type whose subject or object category matches the current
node's category, flipping direction as needed so the sentence reads left-to-right.

Edge types are sampled uniformly (every type gets equal probability per hop, rather
than weighting by real-world edge frequency) so rare predicates still get learned.
Nodes are not repeated within a path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .biolink import predicate_to_name
from .cache import edge_type_key
from .kgx import EdgeType, Node
from .render import render


@dataclass(frozen=True, slots=True)
class Path:
    """One sampled n-hop path, everything the training loop needs.

    `node_ids` has length n_hops+1; `edge_phrases` and `edge_keys` have length n_hops.
    `sentence` is the full-path target string.
    """

    node_ids: list[str]
    edge_phrases: list[str]
    edge_keys: list[str]  # cache keys, direction-specific
    sentence: str


class PathSampler:
    def __init__(
        self,
        nodes: dict[str, Node],
        category_index: dict[str, list[str]],
        edge_types: list[EdgeType],
    ):
        self.nodes = nodes
        self.category_index = category_index
        self.edge_types = edge_types
        # adj[cat] = list of (edge_type_idx, reverse_flag, next_category)
        # An edge type with subject==object contributes twice (once per direction).
        self._adj: dict[str, list[tuple[int, bool, str]]] = {}
        for i, et in enumerate(edge_types):
            self._adj.setdefault(et.subject_category, []).append(
                (i, False, et.object_category)
            )
            if et.object_category != et.subject_category:
                self._adj.setdefault(et.object_category, []).append(
                    (i, True, et.subject_category)
                )
            else:
                # Same-category: reverse is also a valid distinct choice.
                self._adj.setdefault(et.object_category, []).append(
                    (i, True, et.subject_category)
                )

    def sample(self, n_hops: int, rng: random.Random, max_tries: int = 32) -> Path:
        """Sample a single n-hop path. Retries on dead-ends; raises if it can't find one."""
        for _ in range(max_tries):
            path = self._try_sample(n_hops, rng)
            if path is not None:
                return path
        raise RuntimeError(
            f"Could not sample a {n_hops}-hop path after {max_tries} tries "
            "(graph schema too sparse or categories too isolated?)"
        )

    def _try_sample(self, n_hops: int, rng: random.Random) -> Path | None:
        # Pick a starting node from any category that has outgoing edges.
        start_cats = [c for c in self.category_index if c in self._adj]
        if not start_cats:
            return None
        start_cat = rng.choice(start_cats)
        n0 = rng.choice(self.category_index[start_cat])

        node_ids = [n0]
        edge_phrases: list[str] = []
        edge_keys: list[str] = []
        used: set[str] = {n0}
        current_cat = start_cat

        for _ in range(n_hops):
            choices = self._adj.get(current_cat)
            if not choices:
                return None
            et_idx, reverse, next_cat = rng.choice(choices)
            pool = self.category_index.get(next_cat, [])
            # Avoid revisits. For big pools this is basically free.
            tries = 0
            next_id = None
            while tries < 16:
                candidate = rng.choice(pool)
                if candidate not in used:
                    next_id = candidate
                    break
                tries += 1
            if next_id is None:
                return None

            et = self.edge_types[et_idx]
            phrase = render(et.spec, reverse=reverse)
            key = edge_type_key(
                et.subject_category, et.spec, et.object_category, reverse=reverse
            )
            edge_phrases.append(phrase)
            edge_keys.append(key)
            node_ids.append(next_id)
            used.add(next_id)
            current_cat = next_cat

        sentence = self._compose_sentence(node_ids, edge_phrases)
        return Path(
            node_ids=node_ids,
            edge_phrases=edge_phrases,
            edge_keys=edge_keys,
            sentence=sentence,
        )

    def _compose_sentence(self, node_ids: list[str], edge_phrases: list[str]) -> str:
        parts = [self.nodes[node_ids[0]].name]
        for phrase, nid in zip(edge_phrases, node_ids[1:], strict=True):
            parts.append(phrase)
            parts.append(self.nodes[nid].name)
        return " ".join(parts)


# Re-export for convenience so callers don't need to import biolink to make sense of phrases.
__all__ = ["Path", "PathSampler", "predicate_to_name"]
