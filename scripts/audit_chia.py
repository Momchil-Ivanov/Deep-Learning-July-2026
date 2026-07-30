"""Run the reproducible CHIA data-quality audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trial_criteria_ner.data import audit_chia_documents, load_chia_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional local chia_without_scope.zip path.",
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
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_chia_documents(
        archive_path=args.archive,
        cache_dir=args.cache_dir,
    )
    report = audit_chia_documents(documents)
    serialized_report = json.dumps(report, indent=2, sort_keys=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized_report}\n", encoding="utf-8")

    print(serialized_report)


if __name__ == "__main__":
    main()
