"""Run the DistilBERT CUDA, overflow, and checkpoint-resume quality gate."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from trial_criteria_ner.baseline import evaluate_bio_sequences
from trial_criteria_ner.data import MODEL_ENTITY_TYPES, load_chia_documents
from trial_criteria_ner.preprocessing import (
    CriterionExample,
    assign_criteria_to_splits,
    create_nct_splits,
    resolve_criterion_overlaps,
    split_documents_into_criteria,
)
from trial_criteria_ner.transformer_data import (
    ID_TO_LABEL,
    IGNORE_LABEL_ID,
    LABEL_NAMES,
    LABEL_TO_ID,
    TokenClassificationCollator,
    TokenClassificationDataset,
    TransformerWindow,
    audit_transformer_windows,
    tokenize_criteria_windows,
)

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distilbert_smoke_metrics.json"),
        help="JSON quality-gate report.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/distilbert-smoke"),
        help="Ignored smoke checkpoint directory.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-criteria", type=int, default=64)
    parser.add_argument("--validation-criteria", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_label_covering_criteria(
    criteria: Sequence[CriterionExample],
    limit: int,
) -> list[CriterionExample]:
    """Select rare-label examples first, then fill in source order."""

    if limit <= 0:
        raise ValueError("Criterion limit must be positive.")

    selected_indexes: set[int] = set()
    covered_types: set[str] = set()
    for index, criterion in enumerate(criteria):
        criterion_types = {
            entity.entity_type for entity in criterion.entities
        }
        if criterion_types - covered_types:
            selected_indexes.add(index)
            covered_types.update(criterion_types)
        if covered_types == MODEL_ENTITY_TYPES:
            break

    for index in range(len(criteria)):
        if len(selected_indexes) >= limit:
            break
        selected_indexes.add(index)

    if len(selected_indexes) < min(limit, len(criteria)):
        raise RuntimeError("Could not create the requested smoke subset.")
    return [criteria[index] for index in sorted(selected_indexes)[:limit]]


def _windows_for_criteria(
    windows: Sequence[TransformerWindow],
    criteria: Iterable[CriterionExample],
) -> list[TransformerWindow]:
    criterion_ids = {criterion.criterion_id for criterion in criteria}
    return [
        window for window in windows if window.criterion_id in criterion_ids
    ]


def _move_batch(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _next_batch(
    loader: DataLoader[dict[str, Tensor]],
    iterator: object,
) -> tuple[dict[str, Tensor], object]:
    try:
        batch = next(iterator)  # type: ignore[call-overload]
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def train_optimizer_steps(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: AdamW,
    device: torch.device,
    *,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
) -> list[float]:
    """Train a fixed number of optimizer steps and return mean raw losses."""

    model.train()
    iterator = iter(loader)
    losses: list[float] = []

    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        microbatch_losses: list[float] = []
        for _ in range(gradient_accumulation_steps):
            batch, iterator = _next_batch(loader, iterator)
            outputs = model(**_move_batch(batch, device))
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("DistilBERT produced a non-finite loss.")
            microbatch_losses.append(float(loss.detach().cpu()))
            (loss / gradient_accumulation_steps).backward()

        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("DistilBERT produced non-finite gradients.")
        optimizer.step()
        losses.append(sum(microbatch_losses) / len(microbatch_losses))

    return losses


@torch.no_grad()
def model_logits(
    model: PreTrainedModel,
    batch: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    model.eval()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key != "labels"
    }
    return model(**_move_batch(model_inputs, device)).logits.detach().cpu()


@torch.no_grad()
def evaluate_model(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []

    for batch in loader:
        device_batch = _move_batch(batch, device)
        predictions = model(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch["attention_mask"],
        ).logits.argmax(dim=-1)
        labels = device_batch["labels"]

        for gold_row, predicted_row in zip(
            labels, predictions, strict=True
        ):
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

    return evaluate_bio_sequences(gold_sequences, predicted_sequences)


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: object,
    optimizer: AdamW,
    checkpoint_dir: Path,
    *,
    global_step: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)  # type: ignore[attr-defined]
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        checkpoint_dir / "training_state.pt",
    )


def load_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[PreTrainedModel, AdamW, dict[str, object]]:
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_dir
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer_state = torch.load(
        checkpoint_dir / "optimizer.pt",
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)
    training_state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.set_rng_state(training_state["torch_rng_state"])
    torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
    return model, optimizer, training_state


def validate_quality_gate(report: dict[str, object]) -> None:
    checkpoint = report["checkpoint_resume"]
    training = report["training"]
    validation = report["validation_metrics"]
    assert isinstance(checkpoint, dict)
    assert isinstance(training, dict)
    assert isinstance(validation, dict)

    assert report["device"]["type"] == "cuda"  # type: ignore[index]
    assert checkpoint["saved_global_step"] == 2
    assert checkpoint["resumed_global_step"] == 3
    assert checkpoint["maximum_reload_logit_difference"] == 0.0
    assert checkpoint["parameters_changed_after_resume"] is True
    assert all(math.isfinite(loss) for loss in training["losses"])
    assert training["maximum_allocated_bytes"] > 0
    assert validation["relaxed"]["f1"] >= validation["strict"]["f1"]  # type: ignore[index]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; DistilBERT smoke training requires the GPU."
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

    documents = load_chia_documents(cache_dir=args.cache_dir)
    criteria = resolve_criterion_overlaps(
        split_documents_into_criteria(documents)
    )
    splits = create_nct_splits(criterion.nct_id for criterion in criteria)
    assigned = assign_criteria_to_splits(criteria, splits)

    tokenization_started = time.perf_counter()
    all_train_windows = tokenize_criteria_windows(
        assigned["train"],
        tokenizer,
        max_length=args.max_length,
    )
    all_validation_windows = tokenize_criteria_windows(
        assigned["validation"],
        tokenizer,
        max_length=args.max_length,
    )
    tokenization_seconds = time.perf_counter() - tokenization_started

    selected_train_criteria = select_label_covering_criteria(
        assigned["train"],
        args.train_criteria,
    )
    selected_validation_criteria = list(
        assigned["validation"][: args.validation_criteria]
    )
    train_windows = _windows_for_criteria(
        all_train_windows,
        selected_train_criteria,
    )
    validation_windows = _windows_for_criteria(
        all_validation_windows,
        selected_validation_criteria,
    )

    collator = TokenClassificationCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        TokenClassificationDataset(train_windows),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        TokenClassificationDataset(validation_windows),
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        num_labels=len(LABEL_NAMES),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    training_started = time.perf_counter()
    initial_losses = train_optimizer_steps(
        model,
        train_loader,
        optimizer,
        device,
        optimizer_steps=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    reference_batch = next(iter(validation_loader))
    logits_before_save = model_logits(model, reference_batch, device)
    step_two_dir = args.checkpoint_dir / "step-2"
    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        step_two_dir,
        global_step=2,
    )

    del model
    del optimizer
    torch.cuda.empty_cache()

    resumed_model, resumed_optimizer, training_state = load_checkpoint(
        step_two_dir,
        device,
    )
    reloaded_logits = model_logits(
        resumed_model,
        reference_batch,
        device,
    )
    maximum_reload_difference = float(
        (logits_before_save - reloaded_logits).abs().max()
    )

    resumed_losses = train_optimizer_steps(
        resumed_model,
        train_loader,
        resumed_optimizer,
        device,
        optimizer_steps=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    logits_after_resume = model_logits(
        resumed_model,
        reference_batch,
        device,
    )
    parameters_changed = not torch.equal(
        reloaded_logits,
        logits_after_resume,
    )
    step_three_dir = args.checkpoint_dir / "step-3"
    save_checkpoint(
        resumed_model,
        tokenizer,
        resumed_optimizer,
        step_three_dir,
        global_step=3,
    )
    training_seconds = time.perf_counter() - training_started

    validation_metrics = evaluate_model(
        resumed_model,
        validation_loader,
        device,
    )
    report: dict[str, object] = {
        "run_type": "distilbert_smoke",
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_is_fast": tokenizer.is_fast,
        "label_count": len(LABEL_NAMES),
        "labels": list(LABEL_NAMES),
        "max_length": args.max_length,
        "split_seed": splits.seed,
        "test_split_used": False,
        "tokenization_seconds": tokenization_seconds,
        "full_split_window_audit": {
            "train": audit_transformer_windows(all_train_windows),
            "validation": audit_transformer_windows(
                all_validation_windows
            ),
        },
        "smoke_subset": {
            "train_criteria_count": len(selected_train_criteria),
            "train_window_audit": audit_transformer_windows(train_windows),
            "validation_criteria_count": len(
                selected_validation_criteria
            ),
            "validation_window_audit": audit_transformer_windows(
                validation_windows
            ),
        },
        "training": {
            "precision": "float32",
            "batch_size": 1,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "effective_batch_size": args.gradient_accumulation_steps,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "optimizer_steps": 3,
            "losses": initial_losses + resumed_losses,
            "seconds": training_seconds,
            "maximum_allocated_bytes": torch.cuda.max_memory_allocated(),
            "maximum_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "checkpoint_resume": {
            "step_two_directory": str(step_two_dir),
            "step_three_directory": str(step_three_dir),
            "saved_global_step": 2,
            "loaded_global_step": int(training_state["global_step"]),
            "resumed_global_step": 3,
            "maximum_reload_logit_difference": (
                maximum_reload_difference
            ),
            "parameters_changed_after_resume": parameters_changed,
        },
        "validation_metrics": validation_metrics,
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
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

    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{serialized_report}\n", encoding="utf-8")
    print(serialized_report)


if __name__ == "__main__":
    main()
