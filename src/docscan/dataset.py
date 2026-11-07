from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .common import CONDITIONS, DocscanError
from .geometry import load_annotation
from .image_io import load_photo
from .metrics import normalize_text

SOURCES = ("real", "synthetic")
SPLITS = ("dev", "test")
MANIFEST_FIELDS = ("image_id", "doc_id", "image_path", "gt_text_path", "corners_path",
                   "split", "condition", "source", "page_orientation", "notes")


@dataclass(frozen=True)
class Record:
    image_id: str
    doc_id: str
    image_path: Path
    gt_text_path: Path
    corners_path: Path
    split: str
    condition: str
    source: str
    page_orientation: str
    notes: str


def read_manifest(path: Path) -> list[Record]:
    seen_ids, seen_paths, documents = set(), set(), {}
    records = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
                raise DocscanError("invalid_manifest", f"Expected columns: {', '.join(MANIFEST_FIELDS)}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise DocscanError("invalid_manifest", "Duplicate column name")
            for row in reader:
                try:
                    if None in row or any(row[key] is None for key in MANIFEST_FIELDS):
                        raise ValueError("Incorrect number of CSV fields")
                    data = {key: row[key].strip() for key in MANIFEST_FIELDS}
                    for field in ("image_id", "doc_id"):
                        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", data[field]):
                            raise ValueError(f"Invalid {field}: {data[field]!r}")
                    choices = {"split": SPLITS, "condition": CONDITIONS, "source": SOURCES,
                               "page_orientation": ("portrait", "landscape")}
                    for field, allowed in choices.items():
                        if data[field] not in allowed:
                            raise ValueError(f"Invalid {field}: {data[field]!r}")
                    for field in ("image_path", "gt_text_path", "corners_path"):
                        relative = Path(data[field])
                        if not data[field] or relative.anchor:
                            raise ValueError(f"{field} must be a nonempty relative path")
                        data[field] = (path.parent / relative).resolve()
                    record = Record(**data)
                    if record.image_id.casefold() in seen_ids or record.image_path in seen_paths:
                        raise ValueError(f"Duplicate image identifier or path: {record.image_id}")
                    key = record.doc_id.casefold()
                    assignment = (record.split, record.source)
                    if documents.setdefault(key, assignment) != assignment:
                        raise ValueError(f"Document {record.doc_id} crosses split or source")
                    seen_ids.add(record.image_id.casefold())
                    seen_paths.add(record.image_path)
                    records.append(record)
                except ValueError as error:
                    raise DocscanError("invalid_manifest", f"{path}:{reader.line_num}: {error}") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise DocscanError("invalid_manifest", f"Cannot read manifest {path}: {error}") from error
    if not records:
        raise DocscanError("invalid_manifest", f"Manifest has no rows: {path}")
    return records


def identified_source(image_path: Path, requested: str) -> str:
    marker_path = image_path.parent / "dataset.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("source") == "synthetic" and image_path.name in marker.get("images", []):
                if requested == "real":
                    raise DocscanError("source_mismatch", f"{image_path} is recorded as synthetic")
                return "synthetic"
        except (OSError, ValueError, AttributeError, TypeError) as error:
            raise DocscanError("invalid_dataset_marker", f"Cannot read {marker_path}: {error}") from error
    return requested


def reference_text(path: Path) -> str:
    try:
        text = normalize_text(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as error:
        raise DocscanError("ground_truth_read_failed", f"Cannot read {path}: {error}") from error
    if not text:
        raise DocscanError("empty_ground_truth", f"No reference characters in {path}")
    return text


def validate_data(path: Path) -> dict:
    rows = read_manifest(path)
    references, hashes, errors = {}, {}, []
    for row in rows:
        try:
            identified_source(row.image_path, row.source)
            photo = load_photo(row.image_path)
            annotation = load_annotation(row.corners_path, photo)
            if annotation.source is not None and annotation.source != row.source:
                raise DocscanError("source_mismatch", f"Corner annotation source is {annotation.source}")
            text = reference_text(row.gt_text_path)
            if references.setdefault(row.doc_id.casefold(), text) != text:
                raise DocscanError("inconsistent_reference", f"Different transcriptions for {row.doc_id}")
            first_id = hashes.setdefault(photo.sha256, row.image_id)
            if first_id != row.image_id:
                raise DocscanError("duplicate_image_bytes", f"Same image bytes as {first_id}")
        except DocscanError as error:
            errors.append({"image_id": row.image_id, "error_code": error.code, "error": str(error)})
    counts = Counter((row.source, row.split) for row in rows)
    return {"status": "error" if errors else "ok", "manifest": str(path), "images": len(rows),
            "documents": len({row.doc_id.casefold() for row in rows}),
            "counts": {f"{source}/{split}": count for (source, split), count in sorted(counts.items())},
            "errors": errors}
