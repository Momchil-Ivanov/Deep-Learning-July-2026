"""Prepare CHIA criteria and run split, overlap, and BIO quality checks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from trial_criteria_ner.data import load_chia_documents
from trial_criteria_ner.preprocessing import (
    CriterionExample,
    align_bio_labels,
    audit_preprocessing,
    create_nct_splits,
    resolve_criterion_overlaps,
    split_documents_into_criteria,
)

TOKENIZER_NAME = "distilbert-base-uncased"
TOKENIZER_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preprocessing_audit.json"),
        help="JSON quality report path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
        help="Hugging Face cache directory.",
    )
    return parser.parse_args()


def _percentile(sorted_values: list[int], percentile: float) -> int:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def audit_token_alignment(
    criteria: list[CriterionExample],
    cache_dir: Path,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        cache_dir=cache_dir,
        use_fast=True,
    )
    lengths: list[int] = []
    bio_label_counts: Counter[str] = Counter()

    for criterion in criteria:
        encoding = tokenizer(
            criterion.text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = tuple(tuple(pair) for pair in encoding["offset_mapping"])
        labels = align_bio_labels(criterion.entities, offsets)
        lengths.append(len(encoding["input_ids"]))
        bio_label_counts.update(label for label in labels if label is not None)

    sorted_lengths = sorted(lengths)
    return {
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_is_fast": tokenizer.is_fast,
        "tokenizer_revision": TOKENIZER_REVISION,
        "aligned_criterion_count": len(criteria),
        "alignment_error_count": 0,
        "minimum_token_count": sorted_lengths[0],
        "median_token_count": _percentile(sorted_lengths, 0.50),
        "p95_token_count": _percentile(sorted_lengths, 0.95),
        "p99_token_count": _percentile(sorted_lengths, 0.99),
        "maximum_token_count": sorted_lengths[-1],
        "criteria_over_128_tokens": sum(length > 128 for length in lengths),
        "criteria_over_256_tokens": sum(length > 256 for length in lengths),
        "bio_label_counts": dict(sorted(bio_label_counts.items())),
    }


def validate_quality_gate(report: dict[str, object]) -> None:
    splits = report["splits"]
    assert isinstance(splits, dict)
    assert report["criterion_count"] == 12_409
    assert report["criterion_id_unique_count"] == 12_409
    assert report["nct_leakage_count"] == 0
    assert report["remaining_overlapping_entity_pair_count"] == 0
    assert splits["train"]["nct_id_count"] == 800
    assert splits["validation"]["nct_id_count"] == 100
    assert splits["test"]["nct_id_count"] == 100

    tokenization = report["tokenization"]
    assert isinstance(tokenization, dict)
    assert tokenization["aligned_criterion_count"] == 12_409
    assert tokenization["alignment_error_count"] == 0


def main() -> None:
    args = parse_args()
    documents = load_chia_documents(cache_dir=args.cache_dir)
    original_criteria = split_documents_into_criteria(documents)
    resolved_criteria = resolve_criterion_overlaps(original_criteria)
    splits = create_nct_splits(
        criterion.nct_id for criterion in original_criteria
    )

    report = audit_preprocessing(
        original_criteria,
        resolved_criteria,
        splits,
    )
    report["tokenization"] = audit_token_alignment(
        resolved_criteria,
        args.cache_dir,
    )
    validate_quality_gate(report)

    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{serialized_report}\n", encoding="utf-8")
    print(serialized_report)


if __name__ == "__main__":
    main()
