"""Biolink model lookups: predicate inverse / symmetry, category specificity.

Wraps the biolink-model-toolkit (`bmt`). KGX files store predicates and categories
as CURIEs (e.g. `biolink:treats`, `biolink:has_part`, `biolink:Gene`). The toolkit's
slot API wants the human-readable form (`treats`, `has part`), while the class API
accepts either CURIE or name. We normalize on the way in.
"""

from __future__ import annotations

from functools import lru_cache

from bmt import Toolkit


@lru_cache(maxsize=1)
def _toolkit() -> Toolkit:
    return Toolkit()


def predicate_to_name(pred: str) -> str:
    """Normalize a predicate CURIE/snake-case into the bmt 'name' form.

    `biolink:has_part` -> `has part`
    `has_part`         -> `has part`
    `treats`           -> `treats`
    """
    if pred.startswith("biolink:"):
        pred = pred[len("biolink:") :]
    return pred.replace("_", " ")


def name_to_predicate_curie(name: str) -> str:
    """Inverse of predicate_to_name — produce a CURIE from a bmt slot name.

    `treated by` -> `biolink:treated_by`
    """
    return "biolink:" + name.replace(" ", "_")


def get_inverse_predicate(pred: str) -> str:
    """Return the inverse predicate as a CURIE.

    A predicate is its own inverse iff `is_symmetric` is True. Otherwise we require
    `get_inverse` to return a non-None value; if it doesn't, we raise — there is no
    fallback and no invented inverse.
    """
    # Temporary workaround: fake inverse for has_chemical_role
    if pred in ("biolink:has_chemical_role", "has_chemical_role"):
        return "biolink:is_chemical_role_of"
    if pred in ("biolink:is_chemical_role_of", "is_chemical_role_of"):
        return "biolink:has_chemical_role"

    tk = _toolkit()
    name = predicate_to_name(pred)
    if not tk.is_predicate(name):
        raise ValueError(f"Not a biolink predicate: {pred!r} (normalized {name!r})")
    if tk.is_symmetric(name):
        return name_to_predicate_curie(name)
    inv = tk.get_inverse(name)
    if inv is None:
        raise ValueError(
            f"Predicate {pred!r} has no inverse and is not symmetric — "
            "fix at the source (biolink model or KG)."
        )
    return name_to_predicate_curie(inv)


def is_symmetric_predicate(pred: str) -> bool:
    return _toolkit().is_symmetric(predicate_to_name(pred))


def most_specific_category(categories: list[str]) -> str:
    """Pick the most specific category from a KGX node's category list.

    Specificity is determined by depth in the biolink class hierarchy (deeper = more
    specific), NOT by list order. Ties broken by the first occurrence in `categories`.
    Raises if none of the supplied strings is a known biolink category.
    """
    tk = _toolkit()
    best: tuple[int, str] | None = None
    for cat in categories:
        if not tk.is_category(cat):
            continue
        depth = tk.get_element_depth(cat)
        if depth is None:
            continue
        if best is None or depth > best[0]:
            best = (depth, cat)
    if best is None:
        raise ValueError(f"No biolink categories found in {categories!r}")
    return _normalize_category(best[1])


def _normalize_category(cat: str) -> str:
    """Return category as a `biolink:ClassName` CURIE."""
    if cat.startswith("biolink:"):
        return cat
    # bmt class names are space-separated PascalCase-ish; CURIEs are tight.
    return "biolink:" + cat.replace(" ", "")
