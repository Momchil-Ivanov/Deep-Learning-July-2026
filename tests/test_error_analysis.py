"""Tests for entity-level error classification."""

from trial_criteria_ner.error_analysis import classify_sequence_errors


def test_classify_sequence_errors_detects_boundary_and_type() -> None:
    gold = ["B-Condition", "I-Condition", "O", "B-Drug"]
    predicted = ["B-Condition", "O", "O", "B-Condition"]

    counts = classify_sequence_errors(gold, predicted)

    assert counts["boundary_error"] == 1
    assert counts["type_error"] == 1
    assert counts["missed_entity"] == 0
    assert counts["spurious_entity"] == 0
