import pytest

from squashbert.biolink import (
    get_inverse_predicate,
    is_symmetric_predicate,
    most_specific_category,
    predicate_to_name,
)


def test_predicate_to_name_normalizes_curie_and_snake_case():
    assert predicate_to_name("biolink:has_part") == "has part"
    assert predicate_to_name("has_part") == "has part"
    assert predicate_to_name("treats") == "treats"


def test_asymmetric_inverse_round_trip():
    assert get_inverse_predicate("biolink:treats") == "biolink:treated_by"
    assert get_inverse_predicate("biolink:treated_by") == "biolink:treats"
    assert get_inverse_predicate("biolink:has_part") == "biolink:part_of"


def test_symmetric_predicate_is_its_own_inverse():
    assert is_symmetric_predicate("biolink:related_to")
    assert get_inverse_predicate("biolink:related_to") == "biolink:related_to"
    assert get_inverse_predicate("biolink:interacts_with") == "biolink:interacts_with"


def test_unknown_predicate_raises():
    with pytest.raises(ValueError):
        get_inverse_predicate("biolink:not_a_real_predicate_xyz")


def test_most_specific_category_by_hierarchy_not_order():
    # NamedThing is shallower than Gene; order in the list must not matter.
    assert (
        most_specific_category(["biolink:NamedThing", "biolink:Gene", "biolink:BiologicalEntity"])
        == "biolink:Gene"
    )
    assert (
        most_specific_category(["biolink:Gene", "biolink:NamedThing"]) == "biolink:Gene"
    )


def test_most_specific_category_raises_on_empty_or_unknown():
    with pytest.raises(ValueError):
        most_specific_category(["not:a_category", "also:garbage"])
