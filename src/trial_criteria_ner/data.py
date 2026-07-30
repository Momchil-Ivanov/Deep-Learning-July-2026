"""Download, parse, and audit the CHIA BRAT annotations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

CHIA_REPO_ID = "bigbio/chia"
CHIA_ARCHIVE_NAME = "data/chia_without_scope.zip"
CHIA_REVISION = "36e5df0d60dfc5152cd22a807ade73f135105008"

CHIA_SCHEMA_ENTITY_TYPES = frozenset(
    {
        "Condition",
        "Device",
        "Drug",
        "Measurement",
        "Observation",
        "Person",
        "Procedure",
        "Visit",
        "Temporal",
        "Value",
        "Negation",
        "Multiplier",
        "Qualifier",
        "Reference_point",
        "Mood",
    }
)

MODEL_ENTITY_TYPES = frozenset(
    {
        "Condition",
        "Device",
        "Drug",
        "Measurement",
        "Mood",
        "Observation",
        "Person",
        "Pregnancy_considerations",
        "Procedure",
        "Temporal",
        "Value",
    }
)

CriteriaType = Literal["inclusion", "exclusion"]


@dataclass(frozen=True, slots=True)
class ChiaEntity:
    """One BRAT text-bound annotation."""

    annotation_id: str
    entity_type: str
    spans: tuple[tuple[int, int], ...]
    annotated_text: str
    offset_repaired: bool = False


@dataclass(frozen=True, slots=True)
class ChiaDocument:
    """One inclusion or exclusion document from a registered trial."""

    document_id: str
    nct_id: str
    criteria_type: CriteriaType
    text: str
    entities: tuple[ChiaEntity, ...]
    relation_count: int
    equivalence_count: int


def download_chia_archive(cache_dir: str | Path = ".cache/huggingface") -> Path:
    """Download the pinned scope-free CHIA archive and return its local path."""

    archive_path = hf_hub_download(
        repo_id=CHIA_REPO_ID,
        filename=CHIA_ARCHIVE_NAME,
        repo_type="dataset",
        revision=CHIA_REVISION,
        cache_dir=str(cache_dir),
    )
    return Path(archive_path)


def _normalize_newlines(value: str) -> str:
    """Match Python text-mode newline handling for files read from a ZIP."""

    return value.replace("\r\n", "\n").replace("\r", "\n")


def _newline_offset_map(raw_text: str) -> tuple[int, ...]:
    """Map raw CRLF-aware character offsets to normalized LF offsets."""

    offsets = [0] * (len(raw_text) + 1)
    raw_index = 0
    normalized_index = 0

    while raw_index < len(raw_text):
        offsets[raw_index] = normalized_index
        if raw_text.startswith("\r\n", raw_index):
            offsets[raw_index + 1] = normalized_index
            raw_index += 2
        else:
            raw_index += 1
        normalized_index += 1
        offsets[raw_index] = normalized_index

    return tuple(offsets)


def _normalize_entity_offsets(
    entities: tuple[ChiaEntity, ...],
    raw_text: str,
) -> tuple[ChiaEntity, ...]:
    offset_map = _newline_offset_map(raw_text)
    normalized_entities: list[ChiaEntity] = []

    for entity in entities:
        normalized_spans = tuple(
            (
                offset_map[start] if 0 <= start < len(offset_map) else start,
                offset_map[end] if 0 <= end < len(offset_map) else end,
            )
            for start, end in entity.spans
        )
        normalized_entities.append(replace(entity, spans=normalized_spans))

    return tuple(normalized_entities)


def _repair_entity_offsets(
    entities: tuple[ChiaEntity, ...],
    normalized_text: str,
) -> tuple[ChiaEntity, ...]:
    """Repair a mismatch only when its annotated text has one exact location."""

    repaired_entities: list[ChiaEntity] = []
    for entity in entities:
        valid_spans = all(
            0 <= start < end <= len(normalized_text)
            for start, end in entity.spans
        )
        extracted_text = (
            " ".join(normalized_text[start:end] for start, end in entity.spans)
            if valid_spans
            else ""
        )
        if _normalized_whitespace(extracted_text) == _normalized_whitespace(
            entity.annotated_text
        ):
            repaired_entities.append(entity)
            continue

        target = entity.annotated_text.strip()
        first_match = normalized_text.find(target)
        unique_match = (
            target
            and first_match >= 0
            and normalized_text.find(target, first_match + 1) < 0
        )
        if unique_match:
            repaired_entities.append(
                replace(
                    entity,
                    spans=((first_match, first_match + len(target)),),
                    annotated_text=target,
                    offset_repaired=True,
                )
            )
        else:
            repaired_entities.append(entity)

    return tuple(repaired_entities)


def parse_brat_entities(annotation_text: str) -> tuple[ChiaEntity, ...]:
    """Parse text-bound (``T``) annotations from BRAT annotation content."""

    entities: list[ChiaEntity] = []
    last_entity_index: int | None = None

    for line_number, raw_line in enumerate(
        _normalize_newlines(annotation_text).splitlines(), start=1
    ):
        if not raw_line.strip():
            continue

        if "\t" not in raw_line:
            if last_entity_index is not None:
                previous = entities[last_entity_index]
                entities[last_entity_index] = replace(
                    previous,
                    annotated_text=f"{previous.annotated_text}\n{raw_line}",
                )
            continue

        fields = raw_line.split("\t", maxsplit=2)
        annotation_id = fields[0]
        if not annotation_id.startswith("T"):
            last_entity_index = None
            continue

        if len(fields) != 3:
            raise ValueError(
                f"Malformed text annotation on line {line_number}: {raw_line!r}"
            )

        metadata = fields[1].split(maxsplit=1)
        if len(metadata) != 2:
            raise ValueError(
                f"Missing entity offsets on line {line_number}: {raw_line!r}"
            )

        entity_type, span_specification = metadata
        spans: list[tuple[int, int]] = []
        for raw_span in span_specification.split(";"):
            bounds = raw_span.split()
            if len(bounds) != 2:
                raise ValueError(
                    f"Malformed entity span on line {line_number}: {raw_span!r}"
                )
            spans.append((int(bounds[0]), int(bounds[1])))

        entities.append(
            ChiaEntity(
                annotation_id=annotation_id,
                entity_type=entity_type,
                spans=tuple(spans),
                annotated_text=fields[2],
            )
        )
        last_entity_index = len(entities) - 1

    return tuple(entities)


def _annotation_line_counts(annotation_text: str) -> tuple[int, int]:
    lines = _normalize_newlines(annotation_text).splitlines()
    relation_count = sum(line.startswith("R") for line in lines)
    equivalence_count = sum(line.startswith("*") for line in lines)
    return relation_count, equivalence_count


def load_chia_documents(
    archive_path: str | Path | None = None,
    *,
    cache_dir: str | Path = ".cache/huggingface",
) -> list[ChiaDocument]:
    """Load all inclusion and exclusion documents directly from the CHIA ZIP."""

    resolved_archive = (
        Path(archive_path)
        if archive_path is not None
        else download_chia_archive(cache_dir=cache_dir)
    )

    documents: list[ChiaDocument] = []
    with ZipFile(resolved_archive) as archive:
        member_names = set(archive.namelist())
        text_members = sorted(name for name in member_names if name.endswith(".txt"))

        for text_member in text_members:
            document_id = PurePosixPath(text_member).stem
            if document_id.endswith("_inc"):
                criteria_type: CriteriaType = "inclusion"
            elif document_id.endswith("_exc"):
                criteria_type = "exclusion"
            else:
                raise ValueError(f"Unknown CHIA document type: {document_id!r}")

            annotation_member = str(
                PurePosixPath(text_member).with_suffix(".ann")
            )
            if annotation_member not in member_names:
                raise FileNotFoundError(
                    f"Missing annotation file for {text_member!r}"
                )

            raw_text = archive.read(text_member).decode("utf-8-sig")
            text = _normalize_newlines(raw_text)
            annotation_text = _normalize_newlines(
                archive.read(annotation_member).decode("utf-8-sig")
            )
            entities = _normalize_entity_offsets(
                parse_brat_entities(annotation_text),
                raw_text,
            )
            entities = _repair_entity_offsets(entities, text)
            relation_count, equivalence_count = _annotation_line_counts(
                annotation_text
            )

            documents.append(
                ChiaDocument(
                    document_id=document_id,
                    nct_id=document_id.split("_", maxsplit=1)[0],
                    criteria_type=criteria_type,
                    text=text,
                    entities=entities,
                    relation_count=relation_count,
                    equivalence_count=equivalence_count,
                )
            )

    return documents


def _normalized_whitespace(value: str) -> str:
    return " ".join(value.split())


def _entity_text_from_spans(text: str, entity: ChiaEntity) -> str:
    return " ".join(text[start:end] for start, end in entity.spans)


def _spans_overlap(
    first: tuple[int, int], second: tuple[int, int]
) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


def _span_contains(
    outer: tuple[int, int], inner: tuple[int, int]
) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def audit_chia_documents(
    documents: Iterable[ChiaDocument],
) -> dict[str, object]:
    """Return JSON-serializable integrity and distribution statistics."""

    document_list = list(documents)
    entity_type_counts: Counter[str] = Counter()
    raw_entity_type_counts: Counter[str] = Counter()
    criteria_type_counts: Counter[str] = Counter()
    excluded_annotation_type_counts: Counter[str] = Counter()

    empty_document_count = 0
    document_without_entities_count = 0
    nonempty_line_count = 0
    entity_count = 0
    raw_annotation_count = 0
    discontinuous_entity_count = 0
    invalid_span_count = 0
    text_mismatch_count = 0
    duplicate_entity_id_count = 0
    overlapping_entity_pair_count = 0
    nested_entity_pair_count = 0

    for document in document_list:
        criteria_type_counts[document.criteria_type] += 1
        empty_document_count += int(not document.text.strip())
        nonempty_line_count += sum(
            bool(line.strip()) for line in document.text.splitlines()
        )

        annotation_ids = [entity.annotation_id for entity in document.entities]
        duplicate_entity_id_count += len(annotation_ids) - len(set(annotation_ids))
        raw_annotation_count += len(document.entities)
        raw_entity_type_counts.update(
            entity.entity_type for entity in document.entities
        )
        model_entities = tuple(
            entity
            for entity in document.entities
            if entity.entity_type in MODEL_ENTITY_TYPES
        )
        excluded_annotation_type_counts.update(
            entity.entity_type
            for entity in document.entities
            if entity.entity_type not in MODEL_ENTITY_TYPES
        )
        document_without_entities_count += int(not model_entities)

        for entity in model_entities:
            entity_count += 1
            entity_type_counts[entity.entity_type] += 1
            discontinuous_entity_count += int(len(entity.spans) > 1)

            valid_spans = all(
                0 <= start < end <= len(document.text)
                for start, end in entity.spans
            )
            if not valid_spans:
                invalid_span_count += sum(
                    not (0 <= start < end <= len(document.text))
                    for start, end in entity.spans
                )
                continue

            extracted_text = _entity_text_from_spans(document.text, entity)
            text_mismatch_count += int(
                _normalized_whitespace(extracted_text)
                != _normalized_whitespace(entity.annotated_text)
            )

        for index, first in enumerate(model_entities):
            for second in model_entities[index + 1 :]:
                overlapping = any(
                    _spans_overlap(first_span, second_span)
                    for first_span in first.spans
                    for second_span in second.spans
                )
                if not overlapping:
                    continue

                overlapping_entity_pair_count += 1
                nested_entity_pair_count += int(
                    any(
                        _span_contains(first_span, second_span)
                        or _span_contains(second_span, first_span)
                        for first_span in first.spans
                        for second_span in second.spans
                    )
                )

    return {
        "archive_variant": "chia_without_scope",
        "revision": CHIA_REVISION,
        "document_count": len(document_list),
        "nct_id_count": len({document.nct_id for document in document_list}),
        "criteria_type_counts": dict(sorted(criteria_type_counts.items())),
        "nonempty_criteria_line_count": nonempty_line_count,
        "schema_entity_count": sum(
            entity.entity_type in CHIA_SCHEMA_ENTITY_TYPES
            for document in document_list
            for entity in document.entities
        ),
        "schema_entity_type_count": len(CHIA_SCHEMA_ENTITY_TYPES),
        "entity_count": entity_count,
        "entity_type_count": len(entity_type_counts),
        "entity_type_counts": dict(sorted(entity_type_counts.items())),
        "raw_annotation_count": raw_annotation_count,
        "raw_annotation_type_count": len(raw_entity_type_counts),
        "raw_annotation_type_counts": dict(
            sorted(raw_entity_type_counts.items())
        ),
        "excluded_annotation_count": sum(
            excluded_annotation_type_counts.values()
        ),
        "excluded_annotation_type_counts": dict(
            sorted(excluded_annotation_type_counts.items())
        ),
        "relation_count": sum(
            document.relation_count for document in document_list
        ),
        "equivalence_group_count": sum(
            document.equivalence_count for document in document_list
        ),
        "empty_document_count": empty_document_count,
        "document_without_entities_count": document_without_entities_count,
        "duplicate_entity_id_count": duplicate_entity_id_count,
        "discontinuous_entity_count": discontinuous_entity_count,
        "repaired_offset_count": sum(
            entity.offset_repaired
            for document in document_list
            for entity in document.entities
            if entity.entity_type in MODEL_ENTITY_TYPES
        ),
        "invalid_span_count": invalid_span_count,
        "text_mismatch_count": text_mismatch_count,
        "overlapping_entity_pair_count": overlapping_entity_pair_count,
        "nested_entity_pair_count": nested_entity_pair_count,
    }
