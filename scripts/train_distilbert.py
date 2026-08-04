"""Train DistilBERT on the locked CHIA train split with validation early stopping."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from trial_criteria_ner.data import load_chia_documents
from trial_criteria_ner.preprocessing import (
    assign_criteria_to_splits,
    create_nct_splits,
    resolve_criterion_overlaps,
    split_documents_into_criteria,
)
from trial_criteria_ner.transformer_data import (
    LABEL_NAMES,
    LABEL_TO_ID,
    ID_TO_LABEL,
    TokenClassificationCollator,
    TokenClassificationDataset,
    audit_transformer_windows,
    tokenize_criteria_windows,
)
from trial_criteria_ner.transformer_train import (
    build_linear_warmup_scheduler,
    evaluate_model,
    load_training_checkpoint,
    save_training_checkpoint,
    set_reproducible_seed,
    train_one_epoch,
)

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
SEED = 42
CRF_VALIDATION_STRICT_F1 = 0.6339257873401933
CRF_VALIDATION_RELAXED_F1 = 0.753975678203929


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distilbert_training_metrics.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/distilbert"),
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint directory to continue from.",
    )
    return parser.parse_args()


def _optimizer_steps_per_epoch(
    window_count: int,
    gradient_accumulation_steps: int,
) -> int:
    return math.ceil(window_count / gradient_accumulation_steps)


def validate_quality_gate(report: dict[str, object]) -> None:
    assert report["test_split_used"] is False
    assert report["device"]["type"] == "cuda"  # type: ignore[index]
    history = report["epoch_history"]
    assert isinstance(history, list) and history
    for epoch_report in history:
        assert isinstance(epoch_report, dict)
        assert math.isfinite(float(epoch_report["train_mean_loss"]))
        assert 0.0 <= float(epoch_report["validation_strict_f1"]) <= 1.0
        assert float(epoch_report["validation_relaxed_f1"]) >= float(
            epoch_report["validation_strict_f1"]
        )
    best = report["best_checkpoint"]
    assert isinstance(best, dict)
    assert Path(str(best["directory"])).exists()
    assert report["training"]["maximum_allocated_bytes"] > 0  # type: ignore[index]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; full DistilBERT training requires the GPU."
        )

    set_reproducible_seed(SEED)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Offset alignment requires a fast tokenizer.")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("DistilBERT tokenizer has no padding token.")

    print("Loading CHIA and building locked splits...", flush=True)
    documents = load_chia_documents(cache_dir=args.cache_dir)
    criteria = resolve_criterion_overlaps(
        split_documents_into_criteria(documents)
    )
    splits = create_nct_splits(criterion.nct_id for criterion in criteria)
    assigned = assign_criteria_to_splits(criteria, splits)

    print("Tokenizing train/validation windows...", flush=True)
    tokenization_started = time.perf_counter()
    train_windows = tokenize_criteria_windows(
        assigned["train"],
        tokenizer,
        max_length=args.max_length,
    )
    validation_windows = tokenize_criteria_windows(
        assigned["validation"],
        tokenizer,
        max_length=args.max_length,
    )
    tokenization_seconds = time.perf_counter() - tokenization_started

    collator = TokenClassificationCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        TokenClassificationDataset(train_windows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        TokenClassificationDataset(validation_windows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    steps_per_epoch = _optimizer_steps_per_epoch(
        len(train_windows),
        args.gradient_accumulation_steps,
    )
    total_training_steps = steps_per_epoch * args.max_epochs
    warmup_steps = max(1, int(total_training_steps * args.warmup_ratio))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = args.checkpoint_dir / "latest"
    best_dir = args.checkpoint_dir / "best"

    start_epoch = 1
    best_strict_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history: list[dict[str, object]] = []
    global_step = 0

    if args.resume_from is not None:
        print(f"Resuming from {args.resume_from}...", flush=True)
        model, optimizer, scheduler, training_state = load_training_checkpoint(
            args.resume_from,
            device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            total_training_steps=total_training_steps,
            warmup_steps=warmup_steps,
        )
        start_epoch = int(training_state["completed_epoch"]) + 1
        best_strict_f1 = float(training_state["best_strict_f1"])
        best_epoch = int(training_state["best_epoch"])
        epochs_without_improvement = int(
            training_state["epochs_without_improvement"]
        )
        global_step = int(training_state["global_step"])
        history = training_state.get("epoch_history", [])
        assert isinstance(history, list)
        epoch_history = list(history)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
            num_labels=len(LABEL_NAMES),
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = build_linear_warmup_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_training_steps=total_training_steps,
        )

    training_started = time.perf_counter()
    stopped_early = False

    for epoch in range(start_epoch, args.max_epochs + 1):
        print(f"Epoch {epoch}/{args.max_epochs}", flush=True)
        epoch_started = time.perf_counter()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_every=args.log_every,
        )
        global_step += int(train_stats["optimizer_steps"])

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )
        strict = validation_metrics["strict"]
        relaxed = validation_metrics["relaxed"]
        assert isinstance(strict, dict)
        assert isinstance(relaxed, dict)
        strict_f1 = float(strict["f1"])
        relaxed_f1 = float(relaxed["f1"])
        epoch_seconds = time.perf_counter() - epoch_started

        epoch_report = {
            "epoch": epoch,
            "train_mean_loss": float(train_stats["mean_loss"]),
            "train_batch_count": int(train_stats["batch_count"]),
            "optimizer_steps": int(train_stats["optimizer_steps"]),
            "global_step": global_step,
            "validation_strict_f1": strict_f1,
            "validation_relaxed_f1": relaxed_f1,
            "validation_strict": strict,
            "validation_relaxed": relaxed,
            "validation_per_type": validation_metrics["per_type"],
            "seconds": epoch_seconds,
            "maximum_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        epoch_history.append(epoch_report)
        print(
            f"  train_loss={epoch_report['train_mean_loss']:.4f} "
            f"val_strict_f1={strict_f1:.4f} "
            f"val_relaxed_f1={relaxed_f1:.4f} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )

        improved = strict_f1 > best_strict_f1
        if improved:
            best_strict_f1 = strict_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state = {
            "completed_epoch": epoch,
            "global_step": global_step,
            "best_strict_f1": best_strict_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_history": epoch_history,
        }
        save_training_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            latest_dir,
            state=state,
        )
        if improved:
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(latest_dir, best_dir)
            print(f"  saved new best checkpoint to {best_dir}", flush=True)

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            print(
                f"Early stopping after {epoch} epochs "
                f"(patience={args.patience}).",
                flush=True,
            )
            break

    training_seconds = time.perf_counter() - training_started

    print(f"Loading best checkpoint from epoch {best_epoch}...", flush=True)
    best_model = AutoModelForTokenClassification.from_pretrained(
        best_dir
    ).to(device)
    best_validation_metrics = evaluate_model(
        best_model,
        validation_loader,
        device,
    )

    report: dict[str, object] = {
        "run_type": "distilbert_full_training",
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "label_count": len(LABEL_NAMES),
        "labels": list(LABEL_NAMES),
        "max_length": args.max_length,
        "split_seed": splits.seed,
        "test_split_used": False,
        "tokenization_seconds": tokenization_seconds,
        "window_audit": {
            "train": audit_transformer_windows(train_windows),
            "validation": audit_transformer_windows(validation_windows),
        },
        "hyperparameters": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "steps_per_epoch": steps_per_epoch,
            "planned_total_training_steps": total_training_steps,
            "precision": "float32",
        },
        "training": {
            "started_epoch": start_epoch,
            "completed_epochs": len(epoch_history),
            "stopped_early": stopped_early,
            "global_step": global_step,
            "seconds": training_seconds,
            "maximum_allocated_bytes": torch.cuda.max_memory_allocated(),
            "maximum_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "epoch_history": epoch_history,
        "best_checkpoint": {
            "epoch": best_epoch,
            "directory": str(best_dir),
            "validation_strict_f1": best_strict_f1,
            "validation_metrics": best_validation_metrics,
        },
        "comparison_to_crf_validation": {
            "crf_strict_f1": CRF_VALIDATION_STRICT_F1,
            "crf_relaxed_f1": CRF_VALIDATION_RELAXED_F1,
            "distilbert_strict_f1": float(
                best_validation_metrics["strict"]["f1"]  # type: ignore[index]
            ),
            "distilbert_relaxed_f1": float(
                best_validation_metrics["relaxed"]["f1"]  # type: ignore[index]
            ),
            "strict_f1_delta": float(
                best_validation_metrics["strict"]["f1"]  # type: ignore[index]
            )
            - CRF_VALIDATION_STRICT_F1,
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
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
