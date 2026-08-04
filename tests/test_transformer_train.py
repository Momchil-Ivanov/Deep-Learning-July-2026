"""Tests for DistilBERT training helpers."""

from torch.optim import AdamW
from torch.nn import Linear

from trial_criteria_ner.transformer_train import build_linear_warmup_scheduler


def test_linear_warmup_scheduler_peaks_then_decays() -> None:
    optimizer = AdamW(Linear(2, 2).parameters(), lr=1.0)
    scheduler = build_linear_warmup_scheduler(
        optimizer,
        warmup_steps=2,
        total_training_steps=5,
    )

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    peak = max(rates)
    assert peak <= 1.0
    assert rates[0] < peak
    assert rates[-1] < peak
