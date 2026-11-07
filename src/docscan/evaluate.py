from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
import yaml

from .common import (METHODS, DocscanError, elapsed_ms, error_record, file_sha256, finish_run,
                     json_sha256, new_run, read_csv, run_metadata, utc_now, write_csv, write_json)
from .dataset import CONDITIONS, Record, identified_source, read_manifest, reference_text
from .enhance import gray_image, make_variants
from .geometry import corner_overlay, detect_page, load_annotation, warp_document
from .image_io import NORMALIZED_SPACE, WARPED_SPACE, load_photo, write_png
from .metrics import character_errors, normalize_text

KEYS = ("run_id", "image_id", "doc_id", "source", "split", "condition")
PROVENANCE = ("input_sha256", "gt_sha256", "corners_sha256", "input_width", "input_height")
RESULT_FIELDS = KEYS + ("method", "status", "error_code", "error", "S", "D", "I", "N", "cer",
                        "preprocess_ms", "ocr_ms", "total_ms", "variant_width", "variant_height",
                        "model_input_width", "model_input_height", "variant_path", "ocr_json") + PROVENANCE
SUMMARY_FIELDS = ("run_id", "source", "split", "condition", "method", "status", "expected", "measured",
                  "no_text", "errors", "complete_pairs", "coverage", "S", "D", "I", "N", "micro_cer",
                  "mean_preprocess_ms", "mean_ocr_ms", "mean_total_ms", "median_total_ms")
AUTO_FIELDS = KEYS + ("status", "auto_status", "error_code", "error", "candidate_found", "iou", "corner_max_px",
                     "corner_max_diagonal_ratio", "geometry_pass", "text_check", "auto_verified", "manual_rescue",
                     "auto_ms", "auto_overlay", "reference_overlay") + PROVENANCE
AUTO_SUMMARY_FIELDS = ("run_id", "source", "split", "condition", "expected", "evaluated", "candidates",
                       "geometry_passed", "geometry_pass_rate", "text_annotated", "auto_verified",
                       "auto_verified_rate_on_annotated", "manual_rescues", "errors")
OCR_MEASURED = {"ok", "no_text"}


def _keys(record: Record, run_id: str) -> dict:
    return {"run_id": run_id, **{key: getattr(record, key) for key in KEYS[1:]}}


def _path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _annotation(record: Record, photo):
    data = load_annotation(record.corners_path, photo)
    if data.source and data.source != record.source:
        raise DocscanError("source_mismatch", f"Corner annotation source is {data.source}")
    return data


def _shared_rows(record: Record, path: Path, run_id: str, config: dict, engine, references: dict) -> list[dict]:
    from .ocr import save_ocr

    common = _keys(record, run_id)
    folder = path / "pages" / record.image_id
    folder.mkdir(parents=True)
    try:
        identified_source(record.image_path, record.source)
        start = perf_counter()
        photo = load_photo(record.image_path)
        decode_ms = elapsed_ms(start)
        common.update(input_sha256=photo.sha256, input_width=photo.size[0], input_height=photo.size[1])
        reference = reference_text(record.gt_text_path)
        common.update(gt_sha256=file_sha256(record.gt_text_path), N=len(reference))
        if references.setdefault(record.doc_id.casefold(), reference) != reference:
            raise DocscanError("inconsistent_reference", f"Different transcriptions for {record.doc_id}")
        (folder / "gt_normalized.txt").write_text(reference + "\n", encoding="utf-8")
    except (OSError, DocscanError) as error:
        return [{**common, "method": method, "status": "error", **error_record(error)} for method in METHODS]

    variants = {"A": photo.bgr}
    times = {"A": decode_ms}
    geometry_error = None
    try:
        annotation = _annotation(record, photo)
        common["corners_sha256"] = file_sha256(record.corners_path)
        start = perf_counter()
        warped, geometry = warp_document(photo.bgr, annotation.corners, config["geometry"], record.page_orientation)
        gray = gray_image(warped)
        warp_ms = elapsed_ms(start)
        variants["B"] = gray
        times["B"] = round(decode_ms + warp_ms, 3)
        branches, background, branch_ms, settings = make_variants(gray, config["enhance"])
        variants.update(branches)
        times.update({method: round(decode_ms + warp_ms + tick, 3) for method, tick in branch_ms.items()})
        write_json(folder / "geometry.json", {**geometry, "enhancement": settings, "corner_origin": "shared_annotation",
                                                "annotation_sha256": common["corners_sha256"]})
        write_png(folder / "illumination_map.png", background)
    except (OSError, DocscanError, cv2.error, np.linalg.LinAlgError) as error:
        geometry_error = error_record(error)
        write_json(folder / "preprocessing_error.json", geometry_error)

    rows = []
    for method in METHODS:
        row = {**common, "method": method, "status": "error"}
        rows.append(row)
        if method not in variants:
            row.update(geometry_error)
            continue
        try:
            branch = folder / method
            branch.mkdir()
            image = variants[method]
            image_path = branch / "input.png"
            write_png(image_path, image)
            row.update(variant_path=_path(image_path, path), variant_width=image.shape[1], variant_height=image.shape[0],
                       preprocess_ms=times[method])
            result = engine.recognize(image, NORMALIZED_SPACE if method == "A" else WARPED_SPACE)
            save_ocr(branch, result)
            row.update(ocr_json=_path(branch / "ocr.json", path), ocr_ms=result["elapsed_ms"])
            if result["input_resize_ms"] is not None:
                row["preprocess_ms"] = round(row["preprocess_ms"] + result["input_resize_ms"], 3)
                row["model_input_width"], row["model_input_height"] = result["model_input_size"]
            if result["status"] == "error":
                row.update(error_code=result["error_code"], error=result["error"])
                continue
            prediction = normalize_text(result["full_text"])
            (branch / "eval_text.txt").write_text(prediction + "\n" if prediction else "", encoding="utf-8")
            row.update(character_errors(reference, prediction))
            row.update(status=result["status"], total_ms=round(row["preprocess_ms"] + row["ocr_ms"], 3))
        except (OSError, DocscanError, cv2.error, ValueError) as error:
            row.update(**error_record(error))
    return rows


def _conditions(rows: list[dict]):
    if rows:
        yield "ALL", rows
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        if subset:
            yield condition, subset


def summarize_rows(rows: list[dict]) -> list[dict]:
    summaries = []
    for condition, subset in _conditions(rows):
        counts = defaultdict(set)
        ids = {row["image_id"] for row in subset}
        for row in subset:
            if row["status"] in OCR_MEASURED:
                counts[row["image_id"]].add(row["method"])
        paired = sum(methods == set(METHODS) for methods in counts.values())
        for method in METHODS:
            selected = [row for row in subset if row["method"] == method]
            measured = [row for row in selected if row["status"] in OCR_MEASURED]
            stats = {key: sum(row[key] for row in measured) for key in ("S", "D", "I", "N")}
            row = {key: subset[0][key] for key in ("run_id", "source", "split")}
            row.update(stats, condition=condition, method=method, expected=len(ids), measured=len(measured),
                       status="complete" if paired == len(ids) else "incomplete", complete_pairs=paired,
                       no_text=sum(row["status"] == "no_text" for row in measured), errors=len(ids) - len(measured),
                       coverage=len(measured) / len(ids),
                       micro_cer=(stats["S"] + stats["D"] + stats["I"]) / stats["N"] if stats["N"] else None)
            if measured:
                for metric in ("preprocess_ms", "ocr_ms", "total_ms"):
                    row[f"mean_{metric}"] = round(mean(item[metric] for item in measured), 3)
                row["median_total_ms"] = round(median(item["total_ms"] for item in measured), 3)
            summaries.append(row)
    return summaries


def _auto_row(record: Record, path: Path, run_id: str, config: dict) -> dict:
    row = {**_keys(record, run_id), "status": "error", "manual_rescue": "not_attempted"}
    try:
        identified_source(record.image_path, record.source)
        photo = load_photo(record.image_path)
        row.update(input_sha256=photo.sha256, input_width=photo.size[0], input_height=photo.size[1])
        annotation = _annotation(record, photo)
        row["corners_sha256"] = file_sha256(record.corners_path)
        folder = path / "pages" / record.image_id
        folder.mkdir(parents=True)
        tick = perf_counter()
        detection = detect_page(photo.bgr, config["geometry"], record.page_orientation)
        row["auto_ms"] = elapsed_ms(tick)
        corners = detection.corners
        text_marked = bool(annotation.text_polygons)
        row.update(candidate_found=corners is not None, geometry_pass=False, auto_verified=False,
                   text_check="no_candidate" if text_marked else "not_annotated", auto_status="detection_failed")
        if corners is not None:
            ref = annotation.corners
            overlap, _ = cv2.intersectConvexConvex(corners, ref)
            iou = overlap / (cv2.contourArea(corners) + cv2.contourArea(ref) - overlap)
            error = float(np.linalg.norm(corners - ref, axis=1).max())
            ratio = error / float(np.hypot(*photo.size))
            geometry_pass = iou >= config["evaluation"]["auto_iou_min"] and ratio <= config["evaluation"]["auto_corner_error_max"]
            row.update(iou=iou, corner_max_px=error, corner_max_diagonal_ratio=ratio, geometry_pass=geometry_pass)
            if text_marked:
                retained = all(cv2.pointPolygonTest(corners, (float(x), float(y)), False) >= 0
                               for polygon in annotation.text_polygons for x, y in polygon)
                row["text_check"] = "retained" if retained else "clipped"
                row["auto_verified"] = geometry_pass and retained
            row["auto_status"] = ("verified" if row["auto_verified"] else "geometry_only"
                                  if geometry_pass and not text_marked else "below_threshold")
        write_png(folder / "auto_overlay.png", corner_overlay(photo.bgr, corners, "auto" if corners is not None else "detection_failed"))
        write_png(folder / "reference_overlay.png", corner_overlay(photo.bgr, annotation.corners, "reference"))
        row.update(auto_overlay=_path(folder / "auto_overlay.png", path),
                   reference_overlay=_path(folder / "reference_overlay.png", path))
        diagnostics = {"detection": detection.diagnostics, "text_annotation_available": text_marked}
        if row["auto_status"] in {"detection_failed", "below_threshold"}:
            row["manual_rescue"] = "attempted"
            rescued, geometry = warp_document(photo.bgr, annotation.corners, config["geometry"], record.page_orientation)
            write_png(folder / "manual_rescue.png", rescued)
            diagnostics["manual_geometry"] = geometry
            row["manual_rescue"] = "ok"
        write_json(folder / "auto_geometry.json", diagnostics)
        row["status"] = "ok"
    except (DocscanError, OSError, cv2.error, np.linalg.LinAlgError) as error:
        row.update(**error_record(error))
        if row["manual_rescue"] == "attempted":
            row["manual_rescue"] = "error"
    return row


def _summarize_auto(rows: list[dict]) -> list[dict]:
    summaries = []
    for condition, subset in _conditions(rows):
        checked = [row for row in subset if row["status"] == "ok"]
        passed = sum(row["geometry_pass"] for row in checked)
        text_annotated = sum(row["text_check"] != "not_annotated" for row in checked)
        verified = sum(row["auto_verified"] for row in checked)
        summaries.append({**{key: subset[0][key] for key in ("run_id", "source", "split")},
                          "condition": condition, "expected": len(subset), "evaluated": len(checked),
                          "candidates": sum(row["candidate_found"] for row in checked),
                          "geometry_passed": passed, "geometry_pass_rate": passed / len(checked) if checked else None,
                          "text_annotated": text_annotated, "auto_verified": verified,
                          "auto_verified_rate_on_annotated": verified / text_annotated if text_annotated else None,
                          "manual_rescues": sum(row["manual_rescue"] == "ok" for row in checked),
                          "errors": len(subset) - len(checked)})
    return summaries


def freeze_config(dev_run: Path, output: Path, note: str) -> dict:
    run = json.loads((dev_run / "run.json").read_text(encoding="utf-8"))
    if (run.get("command"), run.get("mode"), run.get("split"), run.get("status")) != (
            "benchmark", "shared-corners", "dev", "ok"):
        raise DocscanError("invalid_development_run", "Freeze requires a successful shared-corners dev benchmark")
    config = run["effective_config"]
    models = run["ocr"]["model_file_sha256"]
    if json_sha256(config) != run["config_sha256"] or not models:
        raise DocscanError("invalid_development_run", "Development config or model hashes are missing or inconsistent")
    rows = read_csv(dev_run / "results.csv")
    if not rows or any((row["run_id"], row["split"], row["source"], row["status"] in OCR_MEASURED) !=
                       (run["run_id"], "dev", run["source"], True) for row in rows):
        raise DocscanError("invalid_development_run", "Development result rows do not match the run")
    freeze = {"source": run["source"], "dev_run_id": run["run_id"], "frozen_at_utc": utc_now(),
              "config_sha256": run["config_sha256"], "dev_manifest_sha256": run["manifest_sha256"],
              "dev_input_sha256": run["input_sha256"], "model_file_sha256": models,
              "dev_doc_ids": sorted({row["doc_id"].casefold() for row in rows}),
              "runtime_versions": {name: run["dependencies"][name] for name in ("paddlepaddle", "paddleocr", "paddlex")},
              "selection_note": note}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        yaml.safe_dump({**config, "freeze": freeze}, file, sort_keys=False, allow_unicode=True)
    return {"status": "ok", "config": str(output), **freeze}


def _warm_engine(engine, records: list[Record]) -> dict:
    from .ocr import download_record

    result = {"status": "error", "attempts": []}
    try:
        engine.load()
    except DocscanError as error:
        result.update(**error_record(error))
        return result
    for record in records:
        try:
            identified_source(record.image_path, record.source)
            photo = load_photo(record.image_path)
        except DocscanError as error:
            result["attempts"].append({"image_id": record.image_id, "status": "error", **error_record(error)})
            continue
        warmup = engine.recognize(photo.bgr, NORMALIZED_SPACE)
        result["attempts"].append({"image_id": record.image_id, "status": warmup["status"],
                                   "predict_ms": warmup["elapsed_ms"], "error_code": warmup.get("error_code")})
        result["status"] = warmup["status"]
        if warmup["status"] == "ok":
            break
    result["model_preparation"] = download_record(engine)
    return result


def benchmark(manifest: Path, output: Path, source: str, split: str, mode: str,
              config: dict, freeze: dict | None) -> dict:
    path, run_id = new_run(output)
    run = run_metadata(run_id, config, source)
    run.update(command="benchmark", mode=mode, split=split, status="error", manifest=str(manifest), freeze=freeze,
               manual_corners_used=mode == "shared-corners", manual_fallback_used=False)
    start = perf_counter()
    rows, summaries = [], []
    names = ("results.csv", "summary.csv") if mode == "shared-corners" else ("auto_results.csv", "auto_summary.csv")
    columns = (RESULT_FIELDS, SUMMARY_FIELDS) if mode == "shared-corners" else (AUTO_FIELDS, AUTO_SUMMARY_FIELDS)
    try:
        (path / "config.yaml").write_text(yaml.safe_dump({**config, **({"freeze": freeze} if freeze else {})},
                                                       sort_keys=False, allow_unicode=True), encoding="utf-8")
        run["outputs"]["config"] = "config.yaml"
        records = sorted((row for row in read_manifest(manifest) if (row.source, row.split) == (source, split)),
                         key=lambda row: row.image_id.casefold())
        run["manifest_sha256"] = file_sha256(manifest)
        if not records:
            raise DocscanError("no_selected_data", f"No records for {source}/{split}")
        doc_conditions = defaultdict(set)
        for record in records:
            doc_conditions[record.doc_id.casefold()].add(record.condition)
        planned = {"dev": 4, "test": 6}[split]
        all_conditions = sum(len(conditions) == len(CONDITIONS) for conditions in doc_conditions.values())
        run["dataset"] = {"images": len(records), "documents": len(doc_conditions), "four_condition_docs": all_conditions,
                          "conditions": dict(Counter(row.condition for row in records)),
                          "status": "synthetic_validation" if source == "synthetic" else
                          "plan_met" if all_conditions >= planned else "partial_dataset"}
        if mode == "shared-corners":
            from .ocr import PaddleEngine

            if split == "test" and (not freeze or freeze.get("source") != source or not freeze.get("dev_run_id") or
                                    not freeze.get("model_file_sha256") or freeze.get("config_sha256") != run["config_sha256"]):
                raise DocscanError("config_not_frozen", f"Run a {source}/dev benchmark and freeze its config before test")
            engine = PaddleEngine(config["ocr"])
            if split == "test":
                dev_docs = freeze.get("dev_doc_ids")
                if dev_docs is None:
                    if freeze.get("dev_manifest_sha256") != run["manifest_sha256"]:
                        raise DocscanError("manifest_differs_from_freeze", "This freeze requires the original manifest")
                elif not isinstance(dev_docs, list) or not all(isinstance(doc_id, str) for doc_id in dev_docs):
                    raise DocscanError("invalid_config", "Frozen development IDs must be a list of strings")
                elif set(dev_docs) & doc_conditions.keys():
                    raise DocscanError("document_split_leak", "A test document was present in the frozen development run")
                expected_runtime = {name: run["dependencies"][name] for name in ("paddlepaddle", "paddleocr", "paddlex")}
                if all(expected_runtime.values()) and expected_runtime != freeze.get("runtime_versions"):
                    raise DocscanError("runtime_differs_from_freeze", "Test OCR dependency versions differ from development")
            run["warmup"] = _warm_engine(engine, records)
            run["ocr"] = engine.metadata()
            if split == "test" and engine.model_file_sha256 and engine.model_file_sha256 != freeze["model_file_sha256"]:
                raise DocscanError("models_differ_from_freeze", "Test model files differ from the frozen development run")
            references = {}
            for record in records:
                rows.extend(_shared_rows(record, path, run_id, config, engine, references))
            summaries = summarize_rows(rows)
            successful = sum(row["status"] in OCR_MEASURED for row in rows)
            run.update(rows_expected=len(records) * len(METHODS), rows_measured=successful,
                       complete_pairs=summaries[0]["complete_pairs"],
                       timing_scope={"preprocess_ms": "EXIF decode + shared warp/gray for B-F + branch + model input resize",
                                     "ocr_ms": "PaddleOCR predict after warm-up", "total_ms": "preprocess_ms + ocr_ms",
                                     "excludes": "annotation and reference I/O, exports, CER, download, load and warm-up"})
        else:
            for record in records:
                rows.append(_auto_row(record, path, run_id, config))
            summaries = _summarize_auto(rows)
            successful = sum(row["status"] == "ok" for row in rows)
            run.update(rows_expected=len(records), rows_measured=successful,
                       manual_rescues=summaries[0]["manual_rescues"],
                       manual_corners_used=bool(summaries[0]["manual_rescues"]))
        run["status"] = "ok" if run["rows_expected"] == successful else "incomplete"
        if run["status"] == "incomplete":
            run.update(error_code="incomplete_benchmark", error="At least one input/method could not be evaluated")
    except (DocscanError, OSError, ValueError) as error:
        run.update(**error_record(error))
    finally:
        fingerprints = {row["image_id"]: [row.get(key) for key in PROVENANCE[:3]] for row in rows}
        run["input_sha256"] = json_sha256(fingerprints) if fingerprints else None
        write_csv(path / names[0], columns[0], rows)
        write_csv(path / names[1], columns[1], summaries)
        run["outputs"].update(results=names[0], summary=names[1])
        run["elapsed_ms"]["total_wall"] = elapsed_ms(start)
        finish_run(path, run)
    return {"run_dir": str(path), "run_id": run_id, "status": run["status"], "source": source, "split": split,
            "rows_measured": run.get("rows_measured", 0), "rows_expected": run.get("rows_expected", 0),
            "error_code": run.get("error_code"), "error": run.get("error")}
