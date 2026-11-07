from pathlib import Path

import pytest

from docscan.common import METHODS, DocscanError, json_sha256, read_csv, write_csv
from docscan.config import load_config
from docscan.dataset import MANIFEST_FIELDS, read_manifest, validate_data
from docscan.evaluate import benchmark, summarize_rows
from docscan.metrics import character_errors, normalize_text


@pytest.mark.parametrize("prediction,expected", [
    ("甲乙丙", (0, 0, 0)), ("甲戊丙", (1, 0, 0)), ("甲丙", (0, 1, 0)),
    ("甲丁乙丙", (0, 0, 1)), ("", (0, 3, 0)), ("甲乙丙丁戊己庚", (0, 0, 4)),
])
def test_character_alignment(prediction, expected):
    counts = character_errors("甲乙丙", prediction)
    assert tuple(counts[key] for key in ("S", "D", "I")) == expected
    assert counts["cer"] == sum(expected) / 3


def test_normalization_preserves_case_and_punctuation_and_rejects_empty_truth():
    assert normalize_text(" Ａ，\n中\t 文。 a ") == "A,中文。a"
    with pytest.raises(DocscanError, match="empty"):
        character_errors(normalize_text(" \n"), "x")


def test_summary_micro_cer_and_pair_coverage():
    rows = []
    for image_id, n, substitutions in (("x", 3, 1), ("y", 7, 1)):
        for method in METHODS:
            status = "error" if image_id == "y" and method == "F" else "ok"
            rows.append({"run_id": "test-fixture", "source": "synthetic", "split": "test", "condition": "front_even",
                         "image_id": image_id, "method": method, "status": status,
                         "S": substitutions, "D": 0, "I": 0, "N": n,
                         "preprocess_ms": 2, "ocr_ms": 5, "total_ms": 7})
    aggregate = {row["method"]: row for row in summarize_rows(rows) if row["condition"] == "ALL"}
    assert aggregate["A"]["micro_cer"] == 0.2
    assert aggregate["F"]["coverage"] == 0.5
    assert aggregate["A"]["status"] == "incomplete" and aggregate["A"]["complete_pairs"] == 1


def _write_manifest(samples: Path, destination: Path, rows: list[dict]) -> Path:
    import os

    for row in rows:
        for name in ("image_path", "gt_text_path", "corners_path"):
            row[name] = os.path.relpath(samples / row[name], destination.parent)
    write_csv(destination, MANIFEST_FIELDS, rows)
    return destination


def test_manifest_rejects_split_leak_and_false_real_source(samples, tmp_path):
    source = samples / "manifest.csv"
    rows = read_csv(source)[:2]
    rows[0]["split"] = "test"
    bad = _write_manifest(samples, tmp_path / "leak.csv", rows)
    with pytest.raises(DocscanError, match="crosses split"):
        read_manifest(bad)
    rows = read_csv(source)[:1]
    rows[0]["source"] = "real"
    fake = _write_manifest(samples, tmp_path / "mislabel.csv", rows)
    assert validate_data(fake)["errors"][0]["error_code"] == "source_mismatch"


def test_benchmark_keeps_failed_rows_and_does_not_substitute_sources(samples, tmp_path):
    rows = read_csv(samples / "manifest.csv")[:2]
    rows[1]["image_path"] = "missing.jpg"
    manifest = _write_manifest(samples, tmp_path / "partial.csv", rows)
    config, _ = load_config()
    config["ocr"]["model_dir"] = str(tmp_path / "missing_models")
    output = tmp_path / "incomplete"
    outcome = benchmark(manifest, output, "synthetic", "dev", "shared-corners", config, None)
    assert outcome["status"] == "incomplete" and outcome["rows_expected"] == 12
    entries = read_csv(output / "results.csv")
    assert len(entries) == 12 and all(row["status"] == "error" and row["cer"] == "" for row in entries)
    assert {row["error_code"] for row in entries[6:]} == {"image_read_failed"}
    no_real = benchmark(manifest, tmp_path / "real", "real", "dev", "shared-corners", config, None)
    assert no_real["status"] == "error" and no_real["error_code"] == "no_selected_data"


def test_frozen_development_ids_reject_overlap_across_manifests(samples, tmp_path):
    rows = read_csv(samples / "manifest.csv")[:1]
    rows[0]["split"] = "test"
    manifest = _write_manifest(samples, tmp_path / "test.csv", rows)
    config, _ = load_config()
    freeze = {"source": "synthetic", "config_sha256": json_sha256(config), "dev_run_id": "test-fixture",
              "model_file_sha256": {"fixture": "sha256"}, "dev_doc_ids": [rows[0]["doc_id"]]}
    result = benchmark(manifest, tmp_path / "leak", "synthetic", "test", "shared-corners", config, freeze)
    assert result["status"] == "error" and result["error_code"] == "document_split_leak"
