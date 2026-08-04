"""Entity-level error buckets for qualitative NER analysis."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from trial_criteria_ner.baseline import bio_to_token_entities


def classify_sequence_errors(
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> Counter[str]:
    """Bucket entity-level disagreements for a short error analysis."""

    gold = bio_to_token_entities(gold_labels)
    predicted = bio_to_token_entities(predicted_labels)
    counts: Counter[str] = Counter()
    matched_gold: set[int] = set()
    matched_predicted: set[int] = set()

    for gold_index, gold_entity in enumerate(gold):
        for predicted_index, predicted_entity in enumerate(predicted):
            if predicted_index in matched_predicted:
                continue
            exact = (
                gold_entity.start == predicted_entity.start
                and gold_entity.end == predicted_entity.end
            )
            overlap = min(gold_entity.end, predicted_entity.end) - max(
                gold_entity.start, predicted_entity.start
            )
            if exact and gold_entity.entity_type == predicted_entity.entity_type:
                matched_gold.add(gold_index)
                matched_predicted.add(predicted_index)
                counts["exact_match"] += 1
                break
            if exact and gold_entity.entity_type != predicted_entity.entity_type:
                matched_gold.add(gold_index)
                matched_predicted.add(predicted_index)
                counts["type_error"] += 1
                break
            if (
                overlap > 0
                and gold_entity.entity_type == predicted_entity.entity_type
            ):
                matched_gold.add(gold_index)
                matched_predicted.add(predicted_index)
                counts["boundary_error"] += 1
                break

    counts["missed_entity"] += sum(
        index not in matched_gold for index in range(len(gold))
    )
    counts["spurious_entity"] += sum(
        index not in matched_predicted for index in range(len(predicted))
    )
    return counts


def summarize_error_counts(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for gold, predicted in zip(gold_sequences, predicted_sequences, strict=True):
        totals.update(classify_sequence_errors(gold, predicted))
    return dict(sorted(totals.items()))


def collect_error_examples(
    texts: Sequence[str],
    example_ids: Sequence[str],
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for text, example_id, gold, predicted in zip(
        texts,
        example_ids,
        gold_sequences,
        predicted_sequences,
        strict=True,
    ):
        if list(gold) == list(predicted):
            continue
        counts = classify_sequence_errors(gold, predicted)
        error_kinds = [
            name
            for name in (
                "type_error",
                "boundary_error",
                "missed_entity",
                "spurious_entity",
            )
            if counts[name]
        ]
        if not error_kinds:
            continue
        examples.append(
            {
                "example_id": example_id,
                "text": text[:240],
                "error_kinds": error_kinds,
                "counts": dict(counts),
                "gold_entities": [
                    {
                        "type": entity.entity_type,
                        "start": entity.start,
                        "end": entity.end,
                    }
                    for entity in bio_to_token_entities(gold)
                ],
                "predicted_entities": [
                    {
                        "type": entity.entity_type,
                        "start": entity.start,
                        "end": entity.end,
                    }
                    for entity in bio_to_token_entities(predicted)
                ],
            }
        )
        if len(examples) >= limit:
            break
    return examples
