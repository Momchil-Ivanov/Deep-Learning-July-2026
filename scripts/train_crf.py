"""Train and evaluate the linear-chain CRF baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Sequence

from sklearn_crfsuite import CRF

from trial_criteria_ner.baseline import (
    SequenceExample,
    criteria_to_sequences,
    evaluate_bio_sequences,
    predict_crf,
    train_crf,
)
from trial_criteria_ner.data import load_chia_documents
from trial_criteria_ner.preprocessing import (
    assign_criteria_to_splits,
    create_nct_splits,
    resolve_criterion_overlaps,
    split_documents_into_criteria,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small split subsets and 10 iterations.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override the default 100 full or 10 smoke iterations.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
        help="Hugging Face download cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Metrics JSON path.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Optional pickle output path; defaults to checkpoints for full runs.",
    )
    return parser.parse_args()


def _limited(
    examples: Sequence[SequenceExample],
    limit: int | None,
) -> list[SequenceExample]:
    return list(examples if limit is None else examples[:limit])


def _evaluate_split(
    model: CRF,
    examples: Sequence[SequenceExample],
) -> tuple[dict[str, object], float]:
    prediction_started = time.perf_counter()
    predictions = predict_crf(model, examples)
    prediction_seconds = time.perf_counter() - prediction_started
    gold = [list(example.labels) for example in examples]

    if len(predictions) != len(gold):
        raise RuntimeError("CRF returned the wrong number of sequences.")
    if any(
        len(predicted) != len(expected)
        for predicted, expected in zip(predictions, gold, strict=True)
    ):
        raise RuntimeError("CRF returned the wrong number of token labels.")

    metrics = evaluate_bio_sequences(gold, predictions)
    metrics["by_criteria_type"] = {
        criteria_type: evaluate_bio_sequences(
            [
                gold[index]
                for index, example in enumerate(examples)
                if example.criteria_type == criteria_type
            ],
            [
                predictions[index]
                for index, example in enumerate(examples)
                if example.criteria_type == criteria_type
            ],
        )
        for criteria_type in ("inclusion", "exclusion")
    }
    return metrics, prediction_seconds


def _validate_quality_gate(report: dict[str, object]) -> None:
    results = report["results"]
    assert isinstance(results, dict)
    for split_name in ("validation", "test"):
        split_metrics = results[split_name]
        assert isinstance(split_metrics, dict)
        strict = split_metrics["strict"]
        relaxed = split_metrics["relaxed"]
        assert isinstance(strict, dict)
        assert isinstance(relaxed, dict)
        for metric_name in ("precision", "recall", "f1"):
            assert 0.0 <= strict[metric_name] <= 1.0
            assert 0.0 <= relaxed[metric_name] <= 1.0
        assert relaxed["f1"] >= strict["f1"]


def main() -> None:
    args = parse_args()
    max_iterations = (
        args.max_iterations
        if args.max_iterations is not None
        else (10 if args.smoke_test else 100)
    )
    output_path = args.output or Path(
        "artifacts/crf_smoke_metrics.json"
        if args.smoke_test
        else "artifacts/crf_baseline_metrics.json"
    )
    model_output = args.model_output
    if model_output is None and not args.smoke_test:
        model_output = Path("checkpoints/crf_baseline.pkl")

    documents = load_chia_documents(cache_dir=args.cache_dir)
    criteria = resolve_criterion_overlaps(
        split_documents_into_criteria(documents)
    )
    splits = create_nct_splits(criterion.nct_id for criterion in criteria)
    assigned = assign_criteria_to_splits(criteria, splits)

    sequence_splits = {
        split_name: criteria_to_sequences(split_criteria)
        for split_name, split_criteria in assigned.items()
    }
    limits = (
        {"train": 300, "validation": 100, "test": 100}
        if args.smoke_test
        else {"train": None, "validation": None, "test": None}
    )
    selected = {
        split_name: _limited(sequence_splits[split_name], limit)
        for split_name, limit in limits.items()
    }

    training_started = time.perf_counter()
    model = train_crf(
        selected["train"],
        c1=0.1,
        c2=0.1,
        max_iterations=max_iterations,
    )
    training_seconds = time.perf_counter() - training_started

    validation_metrics, validation_prediction_seconds = _evaluate_split(
        model,
        selected["validation"],
    )
    test_metrics, test_prediction_seconds = _evaluate_split(
        model,
        selected["test"],
    )

    model_size_bytes: int | None = None
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        with model_output.open("wb") as model_file:
            pickle.dump(model, model_file)
        model_size_bytes = model_output.stat().st_size

    labels = sorted(
        {
            label
            for example in selected["train"]
            for label in example.labels
        }
    )
    report: dict[str, object] = {
        "run_type": "smoke" if args.smoke_test else "full",
        "model": "linear_chain_crf",
        "algorithm": "lbfgs",
        "hyperparameters": {
            "c1": 0.1,
            "c2": 0.1,
            "max_iterations": max_iterations,
            "all_possible_transitions": True,
        },
        "split_seed": splits.seed,
        "split_sizes": {
            split_name: {
                "available_nct_id_count": len(getattr(splits, split_name)),
                "available_criterion_count": len(sequence_splits[split_name]),
                "used_nct_id_count": len(
                    {
                        example.nct_id
                        for example in selected[split_name]
                    }
                ),
                "used_criterion_count": len(selected[split_name]),
                "used_token_count": sum(
                    len(example.tokens) for example in selected[split_name]
                ),
            }
            for split_name in ("train", "validation", "test")
        },
        "training_label_count": len(labels),
        "training_labels": labels,
        "model_state_feature_count": len(model.state_features_),
        "model_transition_feature_count": len(model.transition_features_),
        "training_seconds": training_seconds,
        "prediction_seconds": {
            "validation": validation_prediction_seconds,
            "test": test_prediction_seconds,
        },
        "model_output": str(model_output) if model_output is not None else None,
        "model_size_bytes": model_size_bytes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "sklearn_crfsuite": importlib.metadata.version(
                "sklearn-crfsuite"
            ),
            "python_crfsuite": importlib.metadata.version("python-crfsuite"),
        },
        "results": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }
    _validate_quality_gate(report)

    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{serialized_report}\n", encoding="utf-8")
    print(serialized_report)


if __name__ == "__main__":
    main()
