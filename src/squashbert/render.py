"""Render an edge as a natural-language phrase, in either direction.

Two supported patterns (determined empirically from the translator KG):

1. **Plain predicate** — no qualifiers.
   Forward:  `"{predicate}"`           e.g. `"treats"`
   Reverse:  `"{inverse_predicate}"`   e.g. `"treated by"`

2. **Qualified trio** — `qualified_predicate` + `object_aspect_qualifier` +
   `object_direction_qualifier`.
   Forward:  `"{qp} {direction} {aspect} of"`         e.g. `"causes decreased expression of"`
   Reverse:  `"has {direction} {aspect} {qp_inverse}"`     e.g. `"has decreased expression caused by"`
   (the trailing "by" comes from the inverse predicate itself, e.g. `caused by`.)

Context qualifiers (species, anatomical, mechanism, causal_mechanism, disease, onset,
stage, sex, frequency, form_or_variant, generic `qualifier`) are intentionally dropped:
they condition the relation but don't change the predicate itself, and including them
would make the path sentence unwieldy.

Subject-side qualifier trios (`subject_aspect_qualifier`, `subject_direction_qualifier`)
appear in <0.01% of qualified edges and are not supported — they raise
`UnsupportedQualifierPattern`. Fix at the source if this matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .biolink import get_inverse_predicate, predicate_to_name


class UnsupportedQualifierPattern(ValueError):
    """An edge uses a qualifier combination we don't know how to render."""


@dataclass(frozen=True)
class EdgeSpec:
    """The minimal set of fields needed to render an edge phrase.

    For a plain edge: only `predicate`.
    For a qualified edge: `predicate` is the `qualified_predicate` and both
    `object_aspect` and `object_direction` must be set.
    """

    predicate: str  # biolink CURIE
    object_aspect: str | None = None
    object_direction: str | None = None

    def __post_init__(self) -> None:
        # Either the trio is fully present or fully absent — partial is unsupported.
        aspect = self.object_aspect
        direction = self.object_direction
        if (aspect is None) != (direction is None):
            raise UnsupportedQualifierPattern(
                f"object_aspect and object_direction must be both set or both unset "
                f"(got aspect={aspect!r}, direction={direction!r})"
            )


def edge_spec_from_kgx(edge: dict) -> EdgeSpec:
    """Build an EdgeSpec from a KGX edge dict.

    Chooses `qualified_predicate` over `predicate` when present. Raises
    `UnsupportedQualifierPattern` for subject-side qualifier trios.
    """
    if "subject_aspect_qualifier" in edge or "subject_direction_qualifier" in edge:
        raise UnsupportedQualifierPattern(
            "subject-side qualifier trio is not supported; fix at source"
        )
    qp = edge.get("qualified_predicate")
    if qp is not None:
        return EdgeSpec(
            predicate=qp,
            object_aspect=edge.get("object_aspect_qualifier"),
            object_direction=edge.get("object_direction_qualifier"),
        )
    return EdgeSpec(predicate=edge["predicate"])


def render(spec: EdgeSpec, reverse: bool = False) -> str:
    """Produce the natural-language edge phrase."""
    pred_word = predicate_to_name(spec.predicate)
    if spec.object_aspect is None:
        # Plain predicate.
        if reverse:
            return predicate_to_name(get_inverse_predicate(spec.predicate))
        return pred_word
    # Qualified trio.
    aspect = spec.object_aspect.replace("_", " ")
    direction = spec.object_direction.replace("_", " ")
    if reverse:
        inv_word = predicate_to_name(get_inverse_predicate(spec.predicate))
        return f"has {direction} {aspect} {inv_word}"
    return f"{pred_word} {direction} {aspect} of"


def render_pair(spec: EdgeSpec) -> tuple[str, str]:
    """Return (forward, reverse) phrases."""
    return render(spec, reverse=False), render(spec, reverse=True)
