"""One-shot locked test evaluation for CRF and DistilBERT."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from trial_criteria_ner.baseline import (
    SequenceExample,
    criteria_to_sequences,
    evaluate_bio_sequences,
    predict_crf,
)
from trial_criteria_ner.data import load_chia_documents
from trial_criteria_ner.error_analysis import (
    collect_error_examples,
    summarize_error_counts,
)
from trial_criteria_ner.preprocessing import (
    assign_criteria_to_splits,
    create_nct_splits,
    resolve_criterion_overlaps,
    split_documents_into_criteria,
)
from trial_criteria_ner.transformer_data import (
    ID_TO_LABEL,
    IGNORE_LABEL_ID,
    TokenClassificationCollator,
    TokenClassificationDataset,
    TransformerWindow,
    audit_transformer_windows,
    tokenize_criteria_windows,
)
from trial_criteria_ner.transformer_train import move_batch

SEED = 42
TOKENIZER_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
    )
    parser.add_argument(
        "--crf-checkpoint",
        type=Path,
        default=Path("checkpoints/crf_baseline.pkl"),
    )
    parser.add_argument(
        "--distilbert-checkpoint",
        type=Path,
        default=Path("checkpoints/distilbert/best"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/test_evaluation.json"),
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--error-examples", type=int, default=12)
    return parser.parse_args()


def metrics_with_criteria_type(
    gold: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    criteria_types: Sequence[str],
) -> dict[str, object]:
    metrics = evaluate_bio_sequences(gold, predicted)
    metrics["by_criteria_type"] = {
        criteria_type: evaluate_bio_sequences(
            [
                gold[index]
                for index, value in enumerate(criteria_types)
                if value == criteria_type
            ],
            [
                predicted[index]
                for index, value in enumerate(criteria_types)
                if value == criteria_type
            ],
        )
        for criteria_type in ("inclusion", "exclusion")
    }
    return metrics


@torch.no_grad()
def predict_distilbert(
    model: torch.nn.Module,
    windows: Sequence[TransformerWindow],
    *,
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[list[str]], list[list[str]], list[str], list[str]]:
    loader = DataLoader(
        TokenClassificationDataset(windows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TokenClassificationCollator(pad_token_id),
    )
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    criterion_ids: list[str] = []
    criteria_types: list[str] = []
    window_index = 0

    model.eval()
    for batch in loader:
        device_batch = move_batch(batch, device)
        predictions = model(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch["attention_mask"],
        ).logits.argmax(dim=-1)
        labels = device_batch["labels"]

        for gold_row, predicted_row in zip(labels, predictions, strict=True):
            window = windows[window_index]
            mask = gold_row != IGNORE_LABEL_ID
            gold_sequences.append(
                [
                    ID_TO_LABEL[int(label_id)]
                    for label_id in gold_row[mask].detach().cpu()
                ]
            )
            predicted_sequences.append(
                [
                    ID_TO_LABEL[int(label_id)]
                    for label_id in predicted_row[mask].detach().cpu()
                ]
            )
            criterion_ids.append(
                f"{window.criterion_id}#w{window.window_index}"
            )
            criteria_types.append(window.criteria_type)
            window_index += 1

    if window_index != len(windows):
        raise RuntimeError("DistilBERT prediction count mismatch.")
    return gold_sequences, predicted_sequences, criterion_ids, criteria_types


def evaluate_crf_test(
    checkpoint: Path,
    examples: Sequence[SequenceExample],
) -> tuple[dict[str, object], list[list[str]], list[list[str]], float]:
    with checkpoint.open("rb") as handle:
        model = pickle.load(handle)
    started = time.perf_counter()
    predicted = predict_crf(model, examples)
    seconds = time.perf_counter() - started
    gold = [list(example.labels) for example in examples]
    metrics = metrics_with_criteria_type(
        gold,
        predicted,
        [example.criteria_type for example in examples],
    )
    return metrics, gold, predicted, seconds


def validate_quality_gate(report: dict[str, object]) -> None:
    assert report["test_split"]["nct_id_count"] == 100  # type: ignore[index]
    assert report["test_split"]["criterion_count"] == 1310  # type: ignore[index]
    assert report["training_performed"] is False
    for model_name in ("crf", "distilbert"):
        model_report = report[model_name]
        assert isinstance(model_report, dict)
        strict = model_report["metrics"]["strict"]  # type: ignore[index]
        relaxed = model_report["metrics"]["relaxed"]  # type: ignore[index]
        assert isinstance(strict, dict)
        assert isinstance(relaxed, dict)
        assert 0.0 <= float(strict["f1"]) <= 1.0
        assert float(relaxed["f1"]) >= float(strict["f1"])
    comparison = report["comparison"]
    assert isinstance(comparison, dict)
    assert "strict_f1_delta" in comparison


def main() -> None:
    args = parse_args()
    if not args.crf_checkpoint.exists():
        raise FileNotFoundError(f"Missing CRF checkpoint: {args.crf_checkpoint}")
    if not args.distilbert_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing DistilBERT checkpoint: {args.distilbert_checkpoint}"
        )

    print("Loading locked CHIA test split...", flush=True)
    documents = load_chia_documents(cache_dir=args.cache_dir)
    criteria = resolve_criterion_overlaps(
        split_documents_into_criteria(documents)
    )
    splits = create_nct_splits(criterion.nct_id for criterion in criteria)
    assigned = assign_criteria_to_splits(criteria, splits)
    test_criteria = assigned["test"]

    print("Evaluating CRF on locked test set...", flush=True)
    crf_examples = criteria_to_sequences(test_criteria)
    crf_metrics, crf_gold, crf_predicted, crf_seconds = evaluate_crf_test(
        args.crf_checkpoint,
        crf_examples,
    )
    crf_errors = summarize_error_counts(crf_gold, crf_predicted)
    crf_error_examples = collect_error_examples(
        [" ".join(example.tokens) for example in crf_examples],
        [example.criterion_id for example in crf_examples],
        crf_gold,
        crf_predicted,
        limit=args.error_examples,
    )

    print("Evaluating DistilBERT on locked test set...", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DistilBERT test evaluation.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.distilbert_checkpoint,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Checkpoint tokenizer has no pad token.")
    model = AutoModelForTokenClassification.from_pretrained(
        args.distilbert_checkpoint
    ).to(device)

    test_windows = tokenize_criteria_windows(
        test_criteria,
        tokenizer,
        max_length=args.max_length,
    )
    started = time.perf_counter()
    distil_gold, distil_predicted, distil_ids, distil_types = predict_distilbert(
        model,
        test_windows,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    distil_seconds = time.perf_counter() - started
    criterion_text = {
        criterion.criterion_id: criterion.text for criterion in test_criteria
    }
    distil_texts = [
        criterion_text[example_id.split("#w", maxsplit=1)[0]][:240]
        for example_id in distil_ids
    ]
    distil_metrics = metrics_with_criteria_type(
        distil_gold,
        distil_predicted,
        distil_types,
    )
    distil_errors = summarize_error_counts(distil_gold, distil_predicted)
    distil_error_examples = collect_error_examples(
        distil_texts,
        distil_ids,
        distil_gold,
        distil_predicted,
        limit=args.error_examples,
    )

    crf_strict = float(crf_metrics["strict"]["f1"])  # type: ignore[index]
    distil_strict = float(distil_metrics["strict"]["f1"])  # type: ignore[index]
    crf_relaxed = float(crf_metrics["relaxed"]["f1"])  # type: ignore[index]
    distil_relaxed = float(distil_metrics["relaxed"]["f1"])  # type: ignore[index]

    report: dict[str, object] = {
        "run_type": "locked_test_evaluation",
        "seed": SEED,
        "training_performed": False,
        "split_seed": splits.seed,
        "test_split": {
            "nct_id_count": len(splits.test),
            "criterion_count": len(test_criteria),
            "entity_count": sum(
                len(criterion.entities) for criterion in test_criteria
            ),
        },
        "crf": {
            "checkpoint": str(args.crf_checkpoint),
            "example_count": len(crf_examples),
            "prediction_seconds": crf_seconds,
            "metrics": crf_metrics,
            "error_counts": crf_errors,
            "error_examples": crf_error_examples,
        },
        "distilbert": {
            "checkpoint": str(args.distilbert_checkpoint),
            "tokenizer_revision_expected": TOKENIZER_REVISION,
            "window_audit": audit_transformer_windows(test_windows),
            "prediction_seconds": distil_seconds,
            "metrics": distil_metrics,
            "error_counts": distil_errors,
            "error_examples": distil_error_examples,
            "device": torch.cuda.get_device_name(0),
        },
        "comparison": {
            "strict_f1_crf": crf_strict,
            "strict_f1_distilbert": distil_strict,
            "strict_f1_delta": distil_strict - crf_strict,
            "relaxed_f1_crf": crf_relaxed,
            "relaxed_f1_distilbert": distil_relaxed,
            "relaxed_f1_delta": distil_relaxed - crf_relaxed,
            "winner_strict": (
                "distilbert"
                if distil_strict > crf_strict
                else "crf"
                if crf_strict > distil_strict
                else "tie"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
    }
    validate_quality_gate(report)

    serialized = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
