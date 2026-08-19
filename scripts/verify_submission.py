"""Validate the committed exam evidence without retraining either model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trial_criteria_ner.submission import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    return parser.parse_args()


def main() -> None:
    report = validate_submission(parse_args().root)
    print(json.dumps({"status": "ok", **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
