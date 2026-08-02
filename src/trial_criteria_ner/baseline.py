"""Classical CRF baseline for eligibility-criteria NER."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from sklearn_crfsuite import CRF  # type: ignore[import-not-found]

from trial_criteria_ner.preprocessing import (
    CriterionExample,
    align_bio_labels,
    regex_tokenize_with_offsets,
)


@dataclass(frozen=True, slots=True)
class SequenceExample:
    """Tokenized criterion and its aligned BIO labels."""

    criterion_id: str
    nct_id: str
    criteria_type: str
    tokens: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TokenEntity:
    """Entity represented by an inclusive-exclusive token span."""

    entity_type: str
    start: int
    end: int


def criterion_to_sequence(criterion: CriterionExample) -> SequenceExample:
    """Tokenize one criterion and align its character entities to BIO."""

    tokens, offsets = regex_tokenize_with_offsets(criterion.text)
    aligned_labels = align_bio_labels(criterion.entities, offsets)
    if any(label is None for label in aligned_labels):
        raise ValueError(
            f"CRF tokenizer produced a special token for {criterion.criterion_id}."
        )

    return SequenceExample(
        criterion_id=criterion.criterion_id,
        nct_id=criterion.nct_id,
        criteria_type=criterion.criteria_type,
        tokens=tokens,
        offsets=offsets,
        labels=tuple(label for label in aligned_labels if label is not None),
    )


def criteria_to_sequences(
    criteria: Iterable[CriterionExample],
) -> list[SequenceExample]:
    """Convert criterion examples to CRF-ready token sequences."""

    return [criterion_to_sequence(criterion) for criterion in criteria]


def _word_shape(token: str) -> str:
    shape: list[str] = []
    for character in token:
        if character.isupper():
            category = "X"
        elif character.islower():
            category = "x"
        elif character.isdigit():
            category = "d"
        else:
            category = character
        if not shape or shape[-1] != category:
            shape.append(category)
    return "".join(shape)


def token_features(tokens: Sequence[str], index: int) -> dict[str, object]:
    """Create lexical and local-context features for one token."""

    token = tokens[index]
    features: dict[str, object] = {
        "bias": 1.0,
        "word.lower": token.lower(),
        "word.prefix2": token[:2].lower(),
        "word.prefix3": token[:3].lower(),
        "word.suffix2": token[-2:].lower(),
        "word.suffix3": token[-3:].lower(),
        "word.shape": _word_shape(token),
        "word.length": len(token),
        "word.isupper": token.isupper(),
        "word.istitle": token.istitle(),
        "word.isdigit": token.isdigit(),
        "word.hasdigit": any(character.isdigit() for character in token),
    }

    if index == 0:
        features["BOS"] = True
    else:
        previous = tokens[index - 1]
        features.update(
            {
                "-1:word.lower": previous.lower(),
                "-1:word.shape": _word_shape(previous),
                "-1:word.istitle": previous.istitle(),
                "-1:word.isupper": previous.isupper(),
            }
        )

    if index == len(tokens) - 1:
        features["EOS"] = True
    else:
        following = tokens[index + 1]
        features.update(
            {
                "+1:word.lower": following.lower(),
                "+1:word.shape": _word_shape(following),
                "+1:word.istitle": following.istitle(),
                "+1:word.isupper": following.isupper(),
            }
        )

    return features


def sequence_features(example: SequenceExample) -> list[dict[str, object]]:
    """Create CRF features for every token in one sequence."""

    return [
        token_features(example.tokens, index)
        for index in range(len(example.tokens))
    ]


def train_crf(
    examples: Sequence[SequenceExample],
    *,
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
) -> CRF:
    """Fit an L-BFGS linear-chain CRF."""

    model = CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    model.fit(
        [sequence_features(example) for example in examples],
        [list(example.labels) for example in examples],
    )
    return model


def predict_crf(
    model: CRF,
    examples: Sequence[SequenceExample],
) -> list[list[str]]:
    """Predict BIO labels for a collection of examples."""

    predictions = model.predict(
        [sequence_features(example) for example in examples]
    )
    return [list(sequence) for sequence in predictions]


def bio_to_token_entities(labels: Sequence[str]) -> tuple[TokenEntity, ...]:
    """Convert one BIO sequence into token spans.

    An orphan or type-changing ``I`` label starts a new entity, which keeps
    evaluation deterministic even for an invalid model output.
    """

    entities: list[TokenEntity] = []
    current_type: str | None = None
    current_start = 0

    def close_current(end: int) -> None:
        nonlocal current_type
        if current_type is not None:
            entities.append(
                TokenEntity(
                    entity_type=current_type,
                    start=current_start,
                    end=end,
                )
            )
            current_type = None

    for index, label in enumerate(labels):
        if label == "O":
            close_current(index)
            continue

        try:
            prefix, entity_type = label.split("-", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Invalid BIO label: {label!r}") from error
        if prefix not in {"B", "I"}:
            raise ValueError(f"Invalid BIO prefix: {prefix!r}")

        continues_current = prefix == "I" and entity_type == current_type
        if continues_current:
            continue

        close_current(index)
        current_type = entity_type
        current_start = index

    close_current(len(labels))
    return tuple(entities)


def _metric_counts(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    *,
    relaxed: bool,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    overall = {"tp": 0, "fp": 0, "fn": 0}
    by_type: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for gold_labels, predicted_labels in zip(
        gold_sequences, predicted_sequences, strict=True
    ):
        if len(gold_labels) != len(predicted_labels):
            raise ValueError("Gold and predicted token counts differ.")

        gold = bio_to_token_entities(gold_labels)
        predicted = bio_to_token_entities(predicted_labels)
        matched_gold: set[int] = set()
        matched_predicted: set[int] = set()
        candidates: list[tuple[int, int, int]] = []

        for gold_index, gold_entity in enumerate(gold):
            for predicted_index, predicted_entity in enumerate(predicted):
                if gold_entity.entity_type != predicted_entity.entity_type:
                    continue
                overlap = min(gold_entity.end, predicted_entity.end) - max(
                    gold_entity.start, predicted_entity.start
                )
                exact = (
                    gold_entity.start == predicted_entity.start
                    and gold_entity.end == predicted_entity.end
                )
                if (relaxed and overlap > 0) or (not relaxed and exact):
                    candidates.append(
                        (
                            -overlap,
                            gold_index,
                            predicted_index,
                        )
                    )

        for _, gold_index, predicted_index in sorted(candidates):
            if (
                gold_index in matched_gold
                or predicted_index in matched_predicted
            ):
                continue
            matched_gold.add(gold_index)
            matched_predicted.add(predicted_index)
            entity_type = gold[gold_index].entity_type
            overall["tp"] += 1
            by_type[entity_type]["tp"] += 1

        for index, entity in enumerate(gold):
            if index not in matched_gold:
                overall["fn"] += 1
                by_type[entity.entity_type]["fn"] += 1
        for index, entity in enumerate(predicted):
            if index not in matched_predicted:
                overall["fp"] += 1
                by_type[entity.entity_type]["fp"] += 1

    return overall, dict(by_type)


def _scores(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_bio_sequences(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Calculate strict and relaxed entity-level micro scores."""

    strict_counts, strict_by_type = _metric_counts(
        gold_sequences,
        predicted_sequences,
        relaxed=False,
    )
    relaxed_counts, relaxed_by_type = _metric_counts(
        gold_sequences,
        predicted_sequences,
        relaxed=True,
    )
    entity_types = sorted(set(strict_by_type) | set(relaxed_by_type))

    return {
        "strict": _scores(strict_counts),
        "relaxed": _scores(relaxed_counts),
        "per_type": {
            entity_type: {
                "strict": _scores(
                    strict_by_type.get(
                        entity_type,
                        {"tp": 0, "fp": 0, "fn": 0},
                    )
                ),
                "relaxed": _scores(
                    relaxed_by_type.get(
                        entity_type,
                        {"tp": 0, "fp": 0, "fn": 0},
                    )
                ),
            }
            for entity_type in entity_types
        },
    }
