"""Tests for CHIA download-independent parsing and auditing."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from trial_criteria_ner.data import (
    audit_chia_documents,
    load_chia_documents,
    parse_brat_entities,
)


def _write_synthetic_archive(path: Path, *, include_exclusion_ann: bool = True) -> None:
    with ZipFile(path, mode="w") as archive:
        archive.writestr("NCT00000001_inc.txt", "Diabetes\r\n")
        archive.writestr(
            "NCT00000001_inc.ann",
            "T1\tCondition 0 8\tDiabetes\r\n",
        )
        archive.writestr("NCT00000001_exc.txt", "No drug\r\n")
        if include_exclusion_ann:
            archive.writestr(
                "NCT00000001_exc.ann",
                "T1\tNegation 0 2\tNo\r\n"
                "T2\tDrug 3 7\tdrug\r\n"
                "R1\tHas_negation Arg1:T2 Arg2:T1\r\n",
            )


def test_parse_brat_entities_supports_discontinuous_spans() -> None:
    annotation_text = (
        "T1\tCondition 0 5;10 15\talpha gamma\n"
        "R1\tRelated Arg1:T1 Arg2:T2\n"
    )

    entities = parse_brat_entities(annotation_text)

    assert len(entities) == 1
    assert entities[0].annotation_id == "T1"
    assert entities[0].entity_type == "Condition"
    assert entities[0].spans == ((0, 5), (10, 15))
    assert entities[0].annotated_text == "alpha gamma"


def test_load_chia_documents_reads_pairs_and_normalizes_newlines(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "chia.zip"
    _write_synthetic_archive(archive_path)

    documents = load_chia_documents(archive_path)

    assert len(documents) == 2
    assert {document.criteria_type for document in documents} == {
        "inclusion",
        "exclusion",
    }
    assert {document.nct_id for document in documents} == {"NCT00000001"}
    assert all("\r" not in document.text for document in documents)


def test_load_chia_documents_remaps_offsets_after_crlf(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "chia.zip"
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr("NCT00000002_inc.txt", "First\r\nDiabetes\r\n")
        archive.writestr(
            "NCT00000002_inc.ann",
            "T1\tCondition 7 15\tDiabetes\r\n",
        )

    document = load_chia_documents(archive_path)[0]
    entity = document.entities[0]

    assert entity.spans == ((6, 14),)
    assert document.text[6:14] == entity.annotated_text


def test_load_chia_documents_repairs_unique_exact_text_mismatch(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "chia.zip"
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "NCT00000003_inc.txt",
            "Diabetes and hypertension\n",
        )
        archive.writestr(
            "NCT00000003_inc.ann",
            "T1\tCondition 0 25\tDiabetes\n",
        )

    entity = load_chia_documents(archive_path)[0].entities[0]

    assert entity.spans == ((0, 8),)
    assert entity.offset_repaired is True


def test_audit_chia_documents_reports_expected_integrity_counts(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "chia.zip"
    _write_synthetic_archive(archive_path)

    report = audit_chia_documents(load_chia_documents(archive_path))

    assert report["document_count"] == 2
    assert report["nct_id_count"] == 1
    assert report["nonempty_criteria_line_count"] == 2
    assert report["raw_annotation_count"] == 3
    assert report["entity_count"] == 2
    assert report["excluded_annotation_count"] == 1
    assert report["invalid_span_count"] == 0
    assert report["text_mismatch_count"] == 0
    assert report["relation_count"] == 1


def test_load_chia_documents_rejects_missing_annotation_file(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "chia.zip"
    _write_synthetic_archive(archive_path, include_exclusion_ann=False)

    with pytest.raises(FileNotFoundError, match="Missing annotation file"):
        load_chia_documents(archive_path)
