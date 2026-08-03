"""Windowed DistilBERT inputs for CHIA token classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from trial_criteria_ner.data import MODEL_ENTITY_TYPES
from trial_criteria_ner.preprocessing import (
    CriterionExample,
    align_bio_labels,
)

IGNORE_LABEL_ID = -100


def build_label_names() -> tuple[str, ...]:
    """Return the deterministic 23-class BIO label vocabulary."""

    entity_types = sorted(MODEL_ENTITY_TYPES)
    return (
        "O",
        *(f"B-{entity_type}" for entity_type in entity_types),
        *(f"I-{entity_type}" for entity_type in entity_types),
    )


LABEL_NAMES = build_label_names()
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = dict(enumerate(LABEL_NAMES))


@dataclass(frozen=True, slots=True)
class TransformerWindow:
    """One max-length token-classification window."""

    criterion_id: str
    nct_id: str
    criteria_type: str
    window_index: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]


def tokenize_criterion_windows(
    criterion: CriterionExample,
    tokenizer: Any,
    *,
    max_length: int = 128,
) -> list[TransformerWindow]:
    """Tokenize one criterion into non-overlapping overflow windows."""

    encoding = tokenizer(
        criterion.text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=0,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    windows: list[TransformerWindow] = []

    for window_index, input_ids in enumerate(encoding["input_ids"]):
        offsets = tuple(
            (int(start), int(end))
            for start, end in encoding["offset_mapping"][window_index]
        )
        string_labels = align_bio_labels(criterion.entities, offsets)
        label_ids = tuple(
            IGNORE_LABEL_ID if label is None else LABEL_TO_ID[label]
            for label in string_labels
        )
        attention_mask = tuple(
            int(value) for value in encoding["attention_mask"][window_index]
        )
        input_id_tuple = tuple(int(value) for value in input_ids)

        if not (
            len(input_id_tuple)
            == len(attention_mask)
            == len(label_ids)
            == len(offsets)
        ):
            raise ValueError(
                f"Token fields differ in length for {criterion.criterion_id}."
            )
        if len(input_id_tuple) > max_length:
            raise ValueError(
                f"Window exceeds max_length for {criterion.criterion_id}."
            )

        windows.append(
            TransformerWindow(
                criterion_id=criterion.criterion_id,
                nct_id=criterion.nct_id,
                criteria_type=criterion.criteria_type,
                window_index=window_index,
                input_ids=input_id_tuple,
                attention_mask=attention_mask,
                labels=label_ids,
                offsets=offsets,
            )
        )

    if not windows:
        raise ValueError(f"No tokenizer window for {criterion.criterion_id}.")
    return windows


def tokenize_criteria_windows(
    criteria: Iterable[CriterionExample],
    tokenizer: Any,
    *,
    max_length: int = 128,
) -> list[TransformerWindow]:
    """Tokenize criteria while retaining every overflow window."""

    return [
        window
        for criterion in criteria
        for window in tokenize_criterion_windows(
            criterion,
            tokenizer,
            max_length=max_length,
        )
    ]


class TokenClassificationDataset(Dataset[dict[str, Tensor]]):
    """Torch dataset backed by immutable transformer windows."""

    def __init__(self, windows: Sequence[TransformerWindow]) -> None:
        self.windows = tuple(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        window = self.windows[index]
        return {
            "input_ids": torch.tensor(window.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                window.attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(window.labels, dtype=torch.long),
        }


@dataclass(frozen=True, slots=True)
class TokenClassificationCollator:
    """Pad token-classification examples without labeling padding."""

    pad_token_id: int

    def __call__(
        self,
        examples: Sequence[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch.")

        maximum_length = max(
            int(example["input_ids"].shape[0]) for example in examples
        )
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, maximum_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, maximum_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, maximum_length),
            IGNORE_LABEL_ID,
            dtype=torch.long,
        )

        for row, example in enumerate(examples):
            length = int(example["input_ids"].shape[0])
            input_ids[row, :length] = example["input_ids"]
            attention_mask[row, :length] = example["attention_mask"]
            labels[row, :length] = example["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def audit_transformer_windows(
    windows: Sequence[TransformerWindow],
) -> dict[str, object]:
    """Summarize window coverage and BIO labels."""

    criterion_window_counts = Counter(
        window.criterion_id for window in windows
    )
    label_counts: Counter[str] = Counter()
    supervised_token_count = 0

    for window in windows:
        for label_id in window.labels:
            if label_id == IGNORE_LABEL_ID:
                continue
            supervised_token_count += 1
            label_counts[ID_TO_LABEL[label_id]] += 1

    return {
        "window_count": len(windows),
        "criterion_count": len(criterion_window_counts),
        "overflow_criterion_count": sum(
            count > 1 for count in criterion_window_counts.values()
        ),
        "maximum_windows_per_criterion": max(
            criterion_window_counts.values(),
            default=0,
        ),
        "maximum_window_length": max(
            (len(window.input_ids) for window in windows),
            default=0,
        ),
        "supervised_token_count": supervised_token_count,
        "ignored_special_token_count": sum(
            label_id == IGNORE_LABEL_ID
            for window in windows
            for label_id in window.labels
        ),
        "bio_label_counts": dict(sorted(label_counts.items())),
    }
