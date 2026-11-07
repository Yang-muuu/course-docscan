import shutil
from pathlib import Path

import pytest

from docscan.common import METHODS, DocscanError, file_sha256, json_sha256, read_csv, write_csv, write_json
from docscan.config import load_config
from docscan.dataset import MANIFEST_FIELDS, read_manifest, validate_data
from docscan.evaluate import RESULT_FIELDS, benchmark, freeze_config, summarize_rows
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


@pytest.mark.parametrize("separate_manifest", [False, True])
def test_benchmark_rejects_renamed_development_image_before_ocr(samples, tmp_path, monkeypatch, separate_manifest):
    original = read_csv(samples / "manifest.csv")[0]
    duplicate = tmp_path / "renamed.jpg"
    shutil.copyfile(samples / original["image_path"], duplicate)
    test_row = {**original, "split": "test", "image_id": "renamed", "doc_id": "different-id", "image_path": str(duplicate)}
    rows = [test_row] if separate_manifest else [original.copy(), test_row]
    manifest = _write_manifest(samples, tmp_path / "manifest.csv", rows)
    config, _ = load_config()
    freeze = {"source": "synthetic", "config_sha256": json_sha256(config), "dev_run_id": "test-fixture",
              "model_file_sha256": {"fixture": "sha256"}, "dev_doc_ids": [original["doc_id"]],
              "dev_image_sha256": {original["image_id"]: file_sha256(duplicate)}}

    def fail_if_initialized(*args):
        pytest.fail("A leaked test photo must be rejected before initializing OCR")

    monkeypatch.setattr("docscan.ocr.PaddleEngine", fail_if_initialized)
    output = tmp_path / "benchmark"
    result = benchmark(manifest, output, "synthetic", "test", "shared-corners", config, freeze)
    assert result["status"] == "error"
    assert result["error_code"] == ("image_split_leak" if separate_manifest else "duplicate_image_bytes")
    assert not read_csv(output / "results.csv")


@pytest.mark.parametrize("change", ["none", "missing", "duplicate", "other_run", "hash_changed"])
def test_freeze_requires_complete_development_records(tmp_path, change):
    config, _ = load_config()
    development = tmp_path / "development"
    development.mkdir()
    results = development / "results.csv"
    hashes = ["1" * 64, "2" * 64, "3" * 64]
    rows = [{"run_id": "test-fixture", "image_id": "one", "doc_id": "doc",
             "source": "synthetic", "split": "dev", "condition": "front_even",
             "method": method, "status": "ok",
             **dict(zip(("input_sha256", "gt_sha256", "corners_sha256"), hashes, strict=True))} for method in METHODS]
    write_csv(results, RESULT_FIELDS, rows)
    run = {"command": "benchmark", "mode": "shared-corners", "split": "dev", "status": "ok",
           "run_id": "test-fixture", "source": "synthetic", "effective_config": config,
           "config_sha256": json_sha256(config), "manifest_sha256": "4" * 64,
           "input_sha256": json_sha256({"one": hashes}), "rows_expected": 6, "rows_measured": 6, "complete_pairs": 1,
           "ocr": {"model_file_sha256": {"test-fixture": "5" * 64}},
           "dependencies": {name: "test-fixture" for name in ("paddlepaddle", "paddleocr", "paddlex")}}
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "other_run":
        rows[-1]["run_id"] = "another-fixture"
    elif change == "hash_changed":
        run["results_sha256"] = file_sha256(results)
        rows[-1]["cer"] = 0.15
    write_csv(results, RESULT_FIELDS, rows)
    write_json(development / "run.json", run)
    output = tmp_path / "frozen.yaml"
    if change == "none":
        result = freeze_config(development, output, "Test fixture")
        assert result["dev_image_sha256"] == {"one": hashes[0]}
        assert result["dev_results_sha256"] == file_sha256(results)
    else:
        with pytest.raises(DocscanError) as error:
            freeze_config(development, output, "Test fixture")
        assert error.value.code == "invalid_development_run" and not output.exists()


@pytest.mark.parametrize("bad_id", ["review_item.", "Con", "lpt1.scan", "Review_Item"])
def test_manifest_rejects_unsafe_or_colliding_windows_image_ids(samples, tmp_path, bad_id):
    rows = read_csv(samples / "manifest.csv")[:2]
    rows[0]["image_id"], rows[1]["image_id"] = "review_item", bad_id
    manifest = _write_manifest(samples, tmp_path / "windows.csv", rows)
    with pytest.raises(DocscanError) as error:
        read_manifest(manifest)
    assert error.value.code == "invalid_manifest"


def test_real_plan_gate_rejects_synthetic_examples(samples):
    result = validate_data(samples / "manifest.csv", require_real_plan=True)
    assert result["status"] == "error"
    assert result["real_plan"] == {"required_documents": {"dev": 4, "test": 6},
                                   "documents": {"dev": 0, "test": 0}, "ready": False}
    assert {row["error_code"] for row in result["errors"]} == {
        "real_plan_requires_real_source", "real_plan_incomplete",
    }
