"""Shared DistilBERT train, evaluate, and checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, PreTrainedModel

from trial_criteria_ner.baseline import evaluate_bio_sequences
from trial_criteria_ner.transformer_data import ID_TO_LABEL, IGNORE_LABEL_ID


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_model(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, object]:
    """Score entity-level strict/relaxed metrics over supervised tokens."""

    model.eval()
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []

    for batch in loader:
        device_batch = move_batch(batch, device)
        predictions = model(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch["attention_mask"],
        ).logits.argmax(dim=-1)
        labels = device_batch["labels"]

        for gold_row, predicted_row in zip(labels, predictions, strict=True):
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


def train_one_epoch(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    device: torch.device,
    *,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
    log_every: int = 100,
) -> dict[str, float | int]:
    """Run one epoch over the windowed training loader."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    supervised_batches = 0
    optimizer_steps = 0

    for batch_index, batch in enumerate(loader, start=1):
        outputs = model(**move_batch(batch, device))
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("DistilBERT produced a non-finite loss.")

        running_loss += float(loss.detach().cpu())
        supervised_batches += 1
        (loss / gradient_accumulation_steps).backward()

        if batch_index % gradient_accumulation_steps == 0:
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    "DistilBERT produced non-finite gradients."
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            if log_every and optimizer_steps % log_every == 0:
                mean_loss = running_loss / supervised_batches
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"  step={optimizer_steps} "
                    f"mean_loss={mean_loss:.4f} "
                    f"lr={learning_rate:.2e}",
                    flush=True,
                )

    remainder = supervised_batches % gradient_accumulation_steps
    if remainder:
        gradient_norm = clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("DistilBERT produced non-finite gradients.")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    return {
        "mean_loss": running_loss / supervised_batches,
        "batch_count": supervised_batches,
        "optimizer_steps": optimizer_steps,
    }


def save_training_checkpoint(
    model: PreTrainedModel,
    tokenizer: Any,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    checkpoint_dir: Path,
    *,
    state: dict[str, object],
) -> None:
    """Persist model, tokenizer, optimizer, scheduler, and RNG state."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    payload = {
        **state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "python_random_state": random.getstate(),
    }
    torch.save(payload, checkpoint_dir / "training_state.pt")


def load_training_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
    *,
    learning_rate: float,
    weight_decay: float,
    total_training_steps: int,
    warmup_steps: int,
) -> tuple[PreTrainedModel, AdamW, LambdaLR, dict[str, object]]:
    """Restore a full training checkpoint, including the LR schedule."""

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_dir
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.load_state_dict(
        torch.load(
            checkpoint_dir / "optimizer.pt",
            map_location=device,
            weights_only=True,
        )
    )
    scheduler = build_linear_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
    )
    scheduler_path = checkpoint_dir / "scheduler.pt"
    if scheduler_path.exists():
        scheduler.load_state_dict(
            torch.load(
                scheduler_path,
                map_location="cpu",
                weights_only=True,
            )
        )

    training_state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.set_rng_state(training_state["torch_rng_state"])
    if torch.cuda.is_available() and training_state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
    if "python_random_state" in training_state:
        random.setstate(training_state["python_random_state"])
    return model, optimizer, scheduler, training_state


def build_linear_warmup_scheduler(
    optimizer: AdamW,
    *,
    warmup_steps: int,
    total_training_steps: int,
) -> LambdaLR:
    """Create a linear warmup then linear decay schedule."""

    if total_training_steps <= 0:
        raise ValueError("total_training_steps must be positive.")
    warmup_steps = max(0, min(warmup_steps, total_training_steps - 1))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)
