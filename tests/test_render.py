import pytest

from squashbert.render import (
    EdgeSpec,
    UnsupportedQualifierPattern,
    edge_spec_from_kgx,
    render,
    render_pair,
)


def test_plain_predicate_forward_and_reverse():
    spec = EdgeSpec(predicate="biolink:treats")
    fwd, rev = render_pair(spec)
    assert fwd == "treats"
    assert rev == "treated by"


def test_plain_predicate_symmetric_is_own_inverse():
    spec = EdgeSpec(predicate="biolink:related_to")
    fwd, rev = render_pair(spec)
    assert fwd == "related to"
    assert rev == "related to"


def test_qualified_trio_matches_user_example():
    # "causes decreased expression of" / "has decreased expression caused by"
    spec = EdgeSpec(
        predicate="biolink:causes",
        object_aspect="expression",
        object_direction="decreased",
    )
    fwd, rev = render_pair(spec)
    assert fwd == "causes decreased expression of"
    assert rev == "has decreased expression caused by"


def test_qualified_trio_with_underscore_values():
    # Ensure underscore-separated enum values get spaced.
    spec = EdgeSpec(
        predicate="biolink:causes",
        object_aspect="molecular_interaction",
        object_direction="increased",
    )
    fwd = render(spec)
    assert fwd == "causes increased molecular interaction of"


def test_partial_qualifier_trio_rejected():
    with pytest.raises(UnsupportedQualifierPattern):
        EdgeSpec(predicate="biolink:causes", object_aspect="activity")
    with pytest.raises(UnsupportedQualifierPattern):
        EdgeSpec(predicate="biolink:causes", object_direction="decreased")


def test_edge_spec_from_kgx_plain():
    edge = {"predicate": "biolink:treats", "subject": "X", "object": "Y"}
    assert edge_spec_from_kgx(edge) == EdgeSpec(predicate="biolink:treats")


def test_edge_spec_from_kgx_qualified_uses_qualified_predicate():
    edge = {
        "predicate": "biolink:affects",
        "qualified_predicate": "biolink:causes",
        "object_aspect_qualifier": "activity",
        "object_direction_qualifier": "decreased",
        "species_context_qualifier": "NCBITaxon:9606",  # context — must be ignored
        "causal_mechanism_qualifier": "inhibition",  # context — must be ignored
    }
    spec = edge_spec_from_kgx(edge)
    assert spec == EdgeSpec(
        predicate="biolink:causes",
        object_aspect="activity",
        object_direction="decreased",
    )
    assert render(spec) == "causes decreased activity of"
    assert render(spec, reverse=True) == "has decreased activity caused by"


def test_edge_spec_from_kgx_subject_trio_rejected():
    edge = {
        "predicate": "biolink:affects",
        "qualified_predicate": "biolink:causes",
        "subject_aspect_qualifier": "hydrolysis",
        "subject_direction_qualifier": "increased",
    }
    with pytest.raises(UnsupportedQualifierPattern):
        edge_spec_from_kgx(edge)
