"""Offline validation of the files committed for the exam submission."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_ARTIFACTS = (
    "chia_audit.json",
    "preprocessing_audit.json",
    "crf_smoke_metrics.json",
    "crf_baseline_metrics.json",
    "distilbert_smoke_metrics.json",
    "distilbert_training_metrics.json",
    "test_evaluation.json",
)


class SubmissionValidationError(ValueError):
    """Raised when committed submission evidence is missing or inconsistent."""


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Missing required file: {path}")
        return {}
    except json.JSONDecodeError as error:
        errors.append(f"Invalid JSON in {path}: {error}")
        return {}

    if not isinstance(payload, dict):
        errors.append(f"Expected a JSON object in {path}.")
        return {}
    return payload


def _nested(
    payload: Mapping[str, Any],
    *keys: str,
) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def _number(
    payload: Mapping[str, Any],
    path: tuple[str, ...],
    errors: list[str],
    *,
    source: str,
) -> float:
    try:
        value = float(_nested(payload, *path))
    except (KeyError, TypeError, ValueError):
        errors.append(f"Missing numeric field {source}:{'.'.join(path)}.")
        return math.nan
    if not math.isfinite(value):
        errors.append(f"Non-finite value at {source}:{'.'.join(path)}.")
    return value


def _check_close(
    actual: float,
    expected: float,
    label: str,
    errors: list[str],
) -> None:
    if not (
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    ):
        errors.append(
            f"Inconsistent {label}: expected {expected!r}, found {actual!r}."
        )


def _validate_metric_blocks(
    value: Any,
    location: str,
    errors: list[str],
) -> int:
    """Validate every precision/recall/F1 block and strict/relaxed pair."""

    if isinstance(value, list):
        return sum(
            _validate_metric_blocks(item, f"{location}[{index}]", errors)
            for index, item in enumerate(value)
        )
    if not isinstance(value, Mapping):
        return 0

    checked = 0
    metric_keys = {"precision", "recall", "f1"}
    if metric_keys <= value.keys():
        checked += 1
        for key in sorted(metric_keys):
            try:
                score = float(value[key])
            except (TypeError, ValueError):
                errors.append(f"Non-numeric metric at {location}.{key}.")
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                errors.append(
                    f"Metric outside [0, 1] at {location}.{key}: {score!r}."
                )

    strict = value.get("strict")
    relaxed = value.get("relaxed")
    if isinstance(strict, Mapping) and isinstance(relaxed, Mapping):
        if "f1" in strict and "f1" in relaxed:
            try:
                if float(relaxed["f1"]) + 1e-12 < float(strict["f1"]):
                    errors.append(
                        f"Relaxed F1 is below strict F1 at {location}."
                    )
            except (TypeError, ValueError):
                pass

    for key, child in value.items():
        checked += _validate_metric_blocks(child, f"{location}.{key}", errors)
    return checked


def _validate_test_comparison(
    artifacts: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    evaluation = artifacts["test_evaluation.json"]
    baseline = artifacts["crf_baseline_metrics.json"]
    preprocessing = artifacts["preprocessing_audit.json"]

    if evaluation.get("training_performed") is not False:
        errors.append("Locked test evaluation must report training_performed=false.")

    for filename in (
        "distilbert_smoke_metrics.json",
        "distilbert_training_metrics.json",
    ):
        if artifacts[filename].get("test_split_used") is not False:
            errors.append(f"{filename} must report test_split_used=false.")

    for metric_name in ("strict", "relaxed"):
        crf_test = _number(
            evaluation,
            ("crf", "metrics", metric_name, "f1"),
            errors,
            source="test_evaluation.json",
        )
        distilbert_test = _number(
            evaluation,
            ("distilbert", "metrics", metric_name, "f1"),
            errors,
            source="test_evaluation.json",
        )
        comparison_crf = _number(
            evaluation,
            ("comparison", f"{metric_name}_f1_crf"),
            errors,
            source="test_evaluation.json",
        )
        comparison_distilbert = _number(
            evaluation,
            ("comparison", f"{metric_name}_f1_distilbert"),
            errors,
            source="test_evaluation.json",
        )
        comparison_delta = _number(
            evaluation,
            ("comparison", f"{metric_name}_f1_delta"),
            errors,
            source="test_evaluation.json",
        )
        baseline_test = _number(
            baseline,
            ("results", "test", metric_name, "f1"),
            errors,
            source="crf_baseline_metrics.json",
        )

        _check_close(comparison_crf, crf_test, f"{metric_name} CRF F1", errors)
        _check_close(
            comparison_distilbert,
            distilbert_test,
            f"{metric_name} DistilBERT F1",
            errors,
        )
        _check_close(
            comparison_delta,
            distilbert_test - crf_test,
            f"{metric_name} F1 delta",
            errors,
        )
        _check_close(baseline_test, crf_test, f"{metric_name} baseline F1", errors)

    strict_crf = _number(
        evaluation,
        ("comparison", "strict_f1_crf"),
        errors,
        source="test_evaluation.json",
    )
    strict_distilbert = _number(
        evaluation,
        ("comparison", "strict_f1_distilbert"),
        errors,
        source="test_evaluation.json",
    )
    expected_winner = (
        "distilbert" if strict_distilbert > strict_crf else "crf"
    )
    if evaluation.get("comparison", {}).get("winner_strict") != expected_winner:
        errors.append("winner_strict does not match the strict F1 scores.")

    try:
        preprocessing_test = _nested(preprocessing, "splits", "test")
        evaluation_test = _nested(evaluation, "test_split")
        for key in ("criterion_count", "entity_count", "nct_id_count"):
            if preprocessing_test[key] != evaluation_test[key]:
                errors.append(f"Test split mismatch for {key}.")
    except (KeyError, TypeError):
        errors.append("Missing test split summary in preprocessing or evaluation.")


def _validate_notebook(path: Path, errors: list[str]) -> int:
    notebook = _load_json(path, errors)
    if not notebook:
        return 0
    if notebook.get("nbformat") != 4:
        errors.append(f"{path} must use notebook format 4.")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append(f"{path} contains no notebook cells.")
        return 0

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if not code_cells:
        errors.append(f"{path} contains no code cells.")
        return 0

    for index, cell in enumerate(code_cells):
        if cell.get("execution_count") is None:
            errors.append(f"Notebook code cell {index} has not been executed.")
        outputs = cell.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            errors.append(f"Notebook code cell {index} has no saved output.")
    return len(code_cells)


def validate_submission(root: Path) -> dict[str, int]:
    """Validate committed artifacts and notebook without data or checkpoints."""

    root = root.resolve()
    errors: list[str] = []
    artifacts = {
        filename: _load_json(root / "artifacts" / filename, errors)
        for filename in EXPECTED_ARTIFACTS
    }

    metric_block_count = sum(
        _validate_metric_blocks(payload, filename, errors)
        for filename, payload in artifacts.items()
    )
    if all(artifacts.values()):
        _validate_test_comparison(artifacts, errors)

    notebook_code_cell_count = _validate_notebook(
        root / "notebooks" / "TrialCriteriaNER.ipynb",
        errors,
    )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SubmissionValidationError(
            f"Submission validation failed:\n{details}"
        )

    return {
        "artifact_count": len(artifacts),
        "metric_block_count": metric_block_count,
        "notebook_code_cell_count": notebook_code_cell_count,
    }
