"""Regression tests for the committed exam evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from trial_criteria_ner.submission import (
    SubmissionValidationError,
    validate_submission,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _copy_submission(tmp_path: Path) -> Path:
    root = tmp_path / "submission"
    shutil.copytree(REPOSITORY_ROOT / "artifacts", root / "artifacts")
    (root / "notebooks").mkdir(parents=True)
    shutil.copy2(
        REPOSITORY_ROOT / "notebooks" / "TrialCriteriaNER.ipynb",
        root / "notebooks" / "TrialCriteriaNER.ipynb",
    )
    return root


def test_committed_submission_evidence_is_consistent() -> None:
    report = validate_submission(REPOSITORY_ROOT)

    assert report["artifact_count"] == 7
    assert report["metric_block_count"] > 100
    assert report["notebook_code_cell_count"] > 0


def test_submission_validation_rejects_incorrect_f1_delta(
    tmp_path: Path,
) -> None:
    root = _copy_submission(tmp_path)
    evaluation_path = root / "artifacts" / "test_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["comparison"]["strict_f1_delta"] = 0.5
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(
        SubmissionValidationError,
        match="Inconsistent strict F1 delta",
    ):
        validate_submission(root)


def test_submission_validation_rejects_unexecuted_notebook(
    tmp_path: Path,
) -> None:
    root = _copy_submission(tmp_path)
    notebook_path = root / "notebooks" / "TrialCriteriaNER.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    first_code_cell = next(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    first_code_cell["execution_count"] = None
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    with pytest.raises(
        SubmissionValidationError,
        match="has not been executed",
    ):
        validate_submission(root)
