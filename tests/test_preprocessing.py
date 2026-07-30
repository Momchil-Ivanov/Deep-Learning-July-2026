"""Tests for criterion splitting, NCT splits, overlap handling, and BIO labels."""

import pytest

from trial_criteria_ner.data import ChiaDocument, ChiaEntity
from trial_criteria_ner.preprocessing import (
    CriterionExample,
    align_bio_labels,
    assign_criteria_to_splits,
    create_nct_splits,
    regex_tokenize_with_offsets,
    resolve_overlapping_entities,
    split_documents_into_criteria,
)


def _entity(
    annotation_id: str,
    entity_type: str,
    start: int,
    end: int,
    text: str,
) -> ChiaEntity:
    return ChiaEntity(
        annotation_id=annotation_id,
        entity_type=entity_type,
        spans=((start, end),),
        annotated_text=text,
    )


def test_split_documents_into_criteria_rebases_offsets() -> None:
    document = ChiaDocument(
        document_id="NCT00000001_inc",
        nct_id="NCT00000001",
        criteria_type="inclusion",
        text="  Diabetes\nHbA1c above 7%\n",
        entities=(
            _entity("T1", "Condition", 2, 10, "Diabetes"),
            _entity("T2", "Measurement", 11, 16, "HbA1c"),
            _entity("T3", "Value", 23, 25, "7%"),
        ),
        relation_count=0,
        equivalence_count=0,
    )

    criteria = split_documents_into_criteria([document])

    assert [criterion.text for criterion in criteria] == [
        "Diabetes",
        "HbA1c above 7%",
    ]
    assert criteria[0].entities[0].spans == ((0, 8),)
    assert criteria[1].entities[0].spans == ((0, 5),)
    assert criteria[1].entities[1].spans == ((12, 14),)


def test_split_documents_into_criteria_rejects_cross_line_entity() -> None:
    document = ChiaDocument(
        document_id="NCT00000001_inc",
        nct_id="NCT00000001",
        criteria_type="inclusion",
        text="First\nSecond\n",
        entities=(_entity("T1", "Condition", 0, 12, "First Second"),),
        relation_count=0,
        equivalence_count=0,
    )

    with pytest.raises(ValueError, match="crosses a criterion boundary"):
        split_documents_into_criteria([document])


def test_resolve_overlapping_entities_keeps_longest_span() -> None:
    short = _entity("T1", "Condition", 4, 12, "diabetes")
    long = _entity("T2", "Condition", 0, 12, "type 2 diabetes")
    separate = _entity("T3", "Drug", 20, 29, "metformin")

    resolved = resolve_overlapping_entities([short, separate, long])

    assert [entity.annotation_id for entity in resolved] == ["T2", "T3"]


def test_create_nct_splits_is_deterministic_and_leakage_free() -> None:
    nct_ids = [f"NCT{index:08d}" for index in range(10)]

    first = create_nct_splits(
        nct_ids,
        train_count=6,
        validation_count=2,
        seed=7,
    )
    second = create_nct_splits(
        reversed(nct_ids),
        train_count=6,
        validation_count=2,
        seed=7,
    )
    split_sets = first.as_sets()

    assert first == second
    assert len(first.train) == 6
    assert len(first.validation) == 2
    assert len(first.test) == 2
    assert not (split_sets["train"] & split_sets["validation"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["validation"] & split_sets["test"])


def test_assign_criteria_to_splits_uses_parent_nct_id() -> None:
    criteria = [
        CriterionExample(
            criterion_id=f"{nct_id}:1",
            document_id=f"{nct_id}_inc",
            nct_id=nct_id,
            criteria_type="inclusion",
            line_number=1,
            text="Criterion",
            entities=(),
        )
        for nct_id in ("NCT00000001", "NCT00000002", "NCT00000003")
    ]
    splits = create_nct_splits(
        (criterion.nct_id for criterion in criteria),
        train_count=1,
        validation_count=1,
        seed=1,
    )

    assigned = assign_criteria_to_splits(criteria, splits)

    assert sum(len(items) for items in assigned.values()) == 3
    assert all(
        criterion.nct_id in splits.as_sets()[split_name]
        for split_name, items in assigned.items()
        for criterion in items
    )


def test_regex_tokens_align_to_expected_bio_labels() -> None:
    text = "HbA1c above 7%"
    tokens, offsets = regex_tokenize_with_offsets(text)
    entities = (
        _entity("T1", "Measurement", 0, 5, "HbA1c"),
        _entity("T2", "Value", 12, 14, "7%"),
    )

    labels = align_bio_labels(entities, offsets)

    assert tokens == ("HbA1c", "above", "7", "%")
    assert labels == (
        "B-Measurement",
        "O",
        "B-Value",
        "I-Value",
    )


def test_bio_alignment_resolves_token_crossing_adjacent_entities() -> None:
    entities = (
        _entity("T1", "Measurement", 0, 3, "DNA"),
        _entity("T2", "Value", 3, 7, ">126"),
    )
    token_offsets = ((0, 2), (2, 5), (5, 7))

    labels = align_bio_labels(entities, token_offsets)

    assert labels == ("B-Measurement", "B-Value", "I-Value")
