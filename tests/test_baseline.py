"""Tests for CRF features and entity-level evaluation."""

from trial_criteria_ner.baseline import (
    bio_to_token_entities,
    criterion_to_sequence,
    evaluate_bio_sequences,
    token_features,
)
from trial_criteria_ner.data import ChiaEntity
from trial_criteria_ner.preprocessing import CriterionExample


def test_criterion_to_sequence_creates_regex_tokens_and_bio() -> None:
    criterion = CriterionExample(
        criterion_id="NCT00000001_inc:1",
        document_id="NCT00000001_inc",
        nct_id="NCT00000001",
        criteria_type="inclusion",
        line_number=1,
        text="HbA1c above 7%",
        entities=(
            ChiaEntity(
                annotation_id="T1",
                entity_type="Measurement",
                spans=((0, 5),),
                annotated_text="HbA1c",
            ),
            ChiaEntity(
                annotation_id="T2",
                entity_type="Value",
                spans=((12, 14),),
                annotated_text="7%",
            ),
        ),
    )

    sequence = criterion_to_sequence(criterion)

    assert sequence.tokens == ("HbA1c", "above", "7", "%")
    assert sequence.labels == (
        "B-Measurement",
        "O",
        "B-Value",
        "I-Value",
    )


def test_token_features_include_boundaries_and_context() -> None:
    tokens = ("Adult", "patients")

    first = token_features(tokens, 0)
    second = token_features(tokens, 1)

    assert first["BOS"] is True
    assert first["+1:word.lower"] == "patients"
    assert second["EOS"] is True
    assert second["-1:word.lower"] == "adult"


def test_bio_to_token_entities_treats_orphan_i_as_new_entity() -> None:
    entities = bio_to_token_entities(
        ("I-Condition", "I-Condition", "O", "B-Drug")
    )

    assert [(entity.entity_type, entity.start, entity.end) for entity in entities] == [
        ("Condition", 0, 2),
        ("Drug", 3, 4),
    ]


def test_evaluate_bio_sequences_scores_perfect_prediction() -> None:
    labels = [["B-Condition", "I-Condition", "O", "B-Drug"]]

    metrics = evaluate_bio_sequences(labels, labels)

    assert metrics["strict"]["f1"] == 1.0
    assert metrics["relaxed"]["f1"] == 1.0
    assert metrics["strict"]["tp"] == 2


def test_relaxed_metric_matches_partial_same_type_span() -> None:
    gold = [["B-Condition", "I-Condition", "O"]]
    predicted = [["O", "B-Condition", "O"]]

    metrics = evaluate_bio_sequences(gold, predicted)

    assert metrics["strict"]["f1"] == 0.0
    assert metrics["relaxed"]["f1"] == 1.0
    assert metrics["relaxed"]["tp"] == 1
