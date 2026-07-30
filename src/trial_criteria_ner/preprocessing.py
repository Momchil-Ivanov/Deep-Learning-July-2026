"""Criterion splitting, leakage-safe data splits, and BIO alignment."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence

from trial_criteria_ner.data import (
    MODEL_ENTITY_TYPES,
    ChiaDocument,
    ChiaEntity,
    CriteriaType,
)

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class CriterionExample:
    """One non-empty eligibility criterion with criterion-relative entities."""

    criterion_id: str
    document_id: str
    nct_id: str
    criteria_type: CriteriaType
    line_number: int
    text: str
    entities: tuple[ChiaEntity, ...]


@dataclass(frozen=True, slots=True)
class NctSplits:
    """Deterministic, mutually exclusive NCT ID groups."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def as_sets(self) -> dict[SplitName, set[str]]:
        return {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }


def split_documents_into_criteria(
    documents: Iterable[ChiaDocument],
) -> list[CriterionExample]:
    """Split CHIA documents on non-empty lines and rebase entity offsets."""

    criteria: list[CriterionExample] = []

    for document in documents:
        model_entities = tuple(
            entity
            for entity in document.entities
            if entity.entity_type in MODEL_ENTITY_TYPES
        )
        document_offset = 0

        for line_number, line_with_ending in enumerate(
            document.text.splitlines(keepends=True), start=1
        ):
            raw_line = line_with_ending[:-1] if line_with_ending.endswith("\n") else line_with_ending
            leading_whitespace = len(raw_line) - len(raw_line.lstrip())
            criterion_text = raw_line.strip()

            if criterion_text:
                criterion_start = document_offset + leading_whitespace
                criterion_end = criterion_start + len(criterion_text)
                criterion_entities: list[ChiaEntity] = []

                for entity in model_entities:
                    touches_criterion = any(
                        max(start, criterion_start) < min(end, criterion_end)
                        for start, end in entity.spans
                    )
                    entirely_inside = all(
                        criterion_start <= start < end <= criterion_end
                        for start, end in entity.spans
                    )

                    if touches_criterion and not entirely_inside:
                        raise ValueError(
                            "Entity crosses a criterion boundary: "
                            f"{document.document_id}/{entity.annotation_id}"
                        )
                    if entirely_inside:
                        criterion_entities.append(
                            replace(
                                entity,
                                spans=tuple(
                                    (
                                        start - criterion_start,
                                        end - criterion_start,
                                    )
                                    for start, end in entity.spans
                                ),
                            )
                        )

                criteria.append(
                    CriterionExample(
                        criterion_id=f"{document.document_id}:{line_number}",
                        document_id=document.document_id,
                        nct_id=document.nct_id,
                        criteria_type=document.criteria_type,
                        line_number=line_number,
                        text=criterion_text,
                        entities=tuple(criterion_entities),
                    )
                )

            document_offset += len(line_with_ending)

    return criteria


def _entities_overlap(first: ChiaEntity, second: ChiaEntity) -> bool:
    return any(
        max(first_start, second_start) < min(first_end, second_end)
        for first_start, first_end in first.spans
        for second_start, second_end in second.spans
    )


def _entity_envelope_length(entity: ChiaEntity) -> int:
    return max(end for _, end in entity.spans) - min(
        start for start, _ in entity.spans
    )


def _entity_covered_length(entity: ChiaEntity) -> int:
    return sum(end - start for start, end in entity.spans)


def resolve_overlapping_entities(
    entities: Iterable[ChiaEntity],
) -> tuple[ChiaEntity, ...]:
    """Keep the longest entity in each overlap, with deterministic tie-breaks."""

    candidates = sorted(
        entities,
        key=lambda entity: (
            -_entity_envelope_length(entity),
            -_entity_covered_length(entity),
            min(start for start, _ in entity.spans),
            max(end for _, end in entity.spans),
            entity.entity_type,
            entity.annotation_id,
        ),
    )
    selected: list[ChiaEntity] = []
    for entity in candidates:
        if not any(_entities_overlap(entity, kept) for kept in selected):
            selected.append(entity)

    return tuple(
        sorted(
            selected,
            key=lambda entity: (
                min(start for start, _ in entity.spans),
                max(end for _, end in entity.spans),
                entity.entity_type,
                entity.annotation_id,
            ),
        )
    )


def resolve_criterion_overlaps(
    criteria: Iterable[CriterionExample],
) -> list[CriterionExample]:
    """Apply longest-span overlap resolution to every criterion."""

    return [
        replace(
            criterion,
            entities=resolve_overlapping_entities(criterion.entities),
        )
        for criterion in criteria
    ]


def count_with_published_overlap_code(
    criteria: Iterable[CriterionExample],
) -> int:
    """Reproduce the paper repository's span-counting algorithm for comparison.

    The reference code envelopes discontinuous spans and merges adjacent spans.
    This function is an audit comparator only; the merged labels are not used
    for model training because adjacency is not overlap in standard NER.
    """

    total_count = 0
    for criterion in criteria:
        spans = sorted(
            (
                min(start for start, _ in entity.spans),
                max(end for _, end in entity.spans),
                entity.entity_type,
            )
            for entity in criterion.entities
        )
        kept: list[tuple[int, int, str]] = []
        for entity in spans:
            if not kept:
                kept.append(entity)
                continue

            last_kept = kept[-1]
            if entity[0] < last_kept[1]:
                if entity[1] - entity[0] > last_kept[1] - last_kept[0]:
                    kept[-1] = entity
            elif entity[0] == last_kept[1]:
                kept[-1] = (last_kept[0], entity[1], last_kept[2])
            else:
                kept.append(entity)

        total_count += len(kept)

    return total_count


def create_nct_splits(
    nct_ids: Iterable[str],
    *,
    train_count: int = 800,
    validation_count: int = 100,
    seed: int = 42,
) -> NctSplits:
    """Shuffle sorted NCT IDs and create deterministic train/validation/test sets."""

    unique_ids = sorted(set(nct_ids))
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("Train and validation counts must be positive.")
    if train_count + validation_count >= len(unique_ids):
        raise ValueError("At least one NCT ID must remain for the test split.")

    random.Random(seed).shuffle(unique_ids)
    validation_end = train_count + validation_count
    return NctSplits(
        train=tuple(sorted(unique_ids[:train_count])),
        validation=tuple(sorted(unique_ids[train_count:validation_end])),
        test=tuple(sorted(unique_ids[validation_end:])),
        seed=seed,
    )


def assign_criteria_to_splits(
    criteria: Iterable[CriterionExample],
    splits: NctSplits,
) -> dict[SplitName, list[CriterionExample]]:
    """Assign criteria using only their parent trial's NCT ID."""

    split_sets = splits.as_sets()
    id_to_split = {
        nct_id: split_name
        for split_name, ids in split_sets.items()
        for nct_id in ids
    }
    assigned: dict[SplitName, list[CriterionExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for criterion in criteria:
        try:
            split_name = id_to_split[criterion.nct_id]
        except KeyError as error:
            raise ValueError(
                f"NCT ID {criterion.nct_id!r} is missing from the split definition."
            ) from error
        assigned[split_name].append(criterion)

    return assigned


def regex_tokenize_with_offsets(
    text: str,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Tokenize words and punctuation for the CRF baseline."""

    matches = tuple(re.finditer(r"\w+(?:[-']\w+)*|[^\w\s]", text))
    return (
        tuple(match.group() for match in matches),
        tuple(match.span() for match in matches),
    )


def align_bio_labels(
    entities: Iterable[ChiaEntity],
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[str | None, ...]:
    """Align character spans to BIO, resolving tokenizer boundary conflicts.

    A token can cross two adjacent character spans, especially when an unknown
    WordPiece covers punctuation at their boundary. The entity with the largest
    character overlap owns that token; deterministic span-length tie-breaks
    follow.
    """

    ordered_entities = sorted(
        entities,
        key=lambda entity: (
            min(start for start, _ in entity.spans),
            max(end for _, end in entity.spans),
        ),
    )
    owners: list[tuple[ChiaEntity, int] | None] = []
    for token_start, token_end in token_offsets:
        if token_start == token_end:
            owners.append(None)
            continue

        candidates: list[
            tuple[int, int, int, str, int, ChiaEntity]
        ] = []
        for entity in ordered_entities:
            for span_index, (span_start, span_end) in enumerate(entity.spans):
                overlap_length = min(token_end, span_end) - max(
                    token_start, span_start
                )
                if overlap_length <= 0:
                    continue
                candidates.append(
                    (
                        -overlap_length,
                        -(span_end - span_start),
                        span_start,
                        entity.annotation_id,
                        span_index,
                        entity,
                    )
                )

        if not candidates:
            owners.append(None)
            continue

        winner = min(candidates)
        owners.append((winner[-1], winner[-2]))

    labels: list[str | None] = []
    previous_owner: tuple[str, int] | None = None
    for (token_start, token_end), owner in zip(
        token_offsets, owners, strict=True
    ):
        if token_start == token_end:
            labels.append(None)
            previous_owner = None
            continue
        if owner is None:
            labels.append("O")
            previous_owner = None
            continue

        entity, span_index = owner
        owner_key = (entity.annotation_id, span_index)
        prefix = "I" if owner_key == previous_owner else "B"
        labels.append(f"{prefix}-{entity.entity_type}")
        previous_owner = owner_key

    return tuple(labels)


def audit_preprocessing(
    original_criteria: Iterable[CriterionExample],
    resolved_criteria: Iterable[CriterionExample],
    splits: NctSplits,
) -> dict[str, object]:
    """Return JSON-serializable preprocessing and leakage statistics."""

    original = list(original_criteria)
    resolved = list(resolved_criteria)
    if [item.criterion_id for item in original] != [
        item.criterion_id for item in resolved
    ]:
        raise ValueError("Original and resolved criterion order differs.")

    assigned = assign_criteria_to_splits(resolved, splits)
    split_sets = splits.as_sets()
    leakage_count = sum(
        len(split_sets[first] & split_sets[second])
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    remaining_overlap_count = sum(
        _entities_overlap(first, second)
        for criterion in resolved
        for index, first in enumerate(criterion.entities)
        for second in criterion.entities[index + 1 :]
    )
    label_counts = Counter(
        entity.entity_type
        for criterion in resolved
        for entity in criterion.entities
    )

    split_report: dict[str, object] = {}
    for split_name, split_criteria in assigned.items():
        split_report[split_name] = {
            "nct_id_count": len(split_sets[split_name]),
            "criterion_count": len(split_criteria),
            "entity_count": sum(
                len(criterion.entities) for criterion in split_criteria
            ),
        }

    return {
        "seed": splits.seed,
        "criterion_count": len(resolved),
        "criterion_id_unique_count": len(
            {criterion.criterion_id for criterion in resolved}
        ),
        "entity_count_before_overlap_resolution": sum(
            len(criterion.entities) for criterion in original
        ),
        "entity_count_after_overlap_resolution": sum(
            len(criterion.entities) for criterion in resolved
        ),
        "published_code_compatible_entity_count": (
            count_with_published_overlap_code(original)
        ),
        "published_paper_entity_count": 31_944,
        "removed_overlapping_entity_count": sum(
            len(before.entities) - len(after.entities)
            for before, after in zip(original, resolved, strict=True)
        ),
        "remaining_overlapping_entity_pair_count": remaining_overlap_count,
        "criteria_without_entities_count": sum(
            not criterion.entities for criterion in resolved
        ),
        "entity_type_counts": dict(sorted(label_counts.items())),
        "nct_leakage_count": leakage_count,
        "splits": split_report,
    }
