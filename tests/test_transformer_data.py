"""Tests for windowed DistilBERT token-classification inputs."""

import torch

from trial_criteria_ner.data import ChiaEntity
from trial_criteria_ner.preprocessing import CriterionExample
from trial_criteria_ner.transformer_data import (
    IGNORE_LABEL_ID,
    LABEL_NAMES,
    LABEL_TO_ID,
    TokenClassificationCollator,
    TokenClassificationDataset,
    audit_transformer_windows,
    tokenize_criterion_windows,
)


class _TwoWindowTokenizer:
    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "input_ids": [[101, 2001, 102], [101, 2002, 102]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "offset_mapping": [
                [(0, 0), (0, 8), (0, 0)],
                [(0, 0), (9, 15), (0, 0)],
            ],
        }


def _criterion() -> CriterionExample:
    return CriterionExample(
        criterion_id="NCT00000001_inc:1",
        document_id="NCT00000001_inc",
        nct_id="NCT00000001",
        criteria_type="inclusion",
        line_number=1,
        text="diabetes adults",
        entities=(
            ChiaEntity(
                annotation_id="T1",
                entity_type="Condition",
                spans=((0, 8),),
                annotated_text="diabetes",
            ),
        ),
    )


def test_label_vocabulary_has_o_and_22_bio_labels() -> None:
    assert len(LABEL_NAMES) == 23
    assert LABEL_NAMES[0] == "O"
    assert set(LABEL_TO_ID) == set(LABEL_NAMES)
    assert "B-Condition" in LABEL_TO_ID
    assert "I-Pregnancy_considerations" in LABEL_TO_ID


def test_tokenize_criterion_retains_all_overflow_windows() -> None:
    windows = tokenize_criterion_windows(
        _criterion(),
        _TwoWindowTokenizer(),
        max_length=3,
    )

    assert len(windows) == 2
    assert windows[0].labels == (
        IGNORE_LABEL_ID,
        LABEL_TO_ID["B-Condition"],
        IGNORE_LABEL_ID,
    )
    assert windows[1].labels == (
        IGNORE_LABEL_ID,
        LABEL_TO_ID["O"],
        IGNORE_LABEL_ID,
    )


def test_dataset_and_collator_pad_with_ignored_labels() -> None:
    windows = tokenize_criterion_windows(
        _criterion(),
        _TwoWindowTokenizer(),
        max_length=3,
    )
    dataset = TokenClassificationDataset(windows)
    shorter = {
        key: value[:2]
        for key, value in dataset[0].items()
    }
    batch = TokenClassificationCollator(pad_token_id=0)(
        [dataset[1], shorter]
    )

    assert batch["input_ids"].shape == torch.Size((2, 3))
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][1, 2].item() == IGNORE_LABEL_ID


def test_transformer_window_audit_reports_overflow() -> None:
    windows = tokenize_criterion_windows(
        _criterion(),
        _TwoWindowTokenizer(),
        max_length=3,
    )

    report = audit_transformer_windows(windows)

    assert report["window_count"] == 2
    assert report["criterion_count"] == 1
    assert report["overflow_criterion_count"] == 1
    assert report["supervised_token_count"] == 2
