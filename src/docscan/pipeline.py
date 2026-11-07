from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .common import DocscanError, elapsed_ms, error_record, file_sha256, finish_run, json_sha256, new_run, run_metadata, write_json
from .dataset import identified_source
from .enhance import gray_image, make_variants
from .geometry import corner_overlay, detect_page, load_annotation, warp_document
from .image_io import IMAGE_SUFFIXES, NORMALIZED_SPACE, WARPED_SPACE, load_photo, write_png

VARIANT_FILES = {"A": "input_normalized.png", "B": "warped_gray.png", "C": "binary_otsu.png",
                 "D": "binary_adaptive.png", "E": "enhanced_gray.png", "F": "binary_enhanced.png"}


def scan_one(input_path: Path, corners_path: Path | None, path: Path, run_id: str,
             config: dict, options: dict, engine) -> dict:
    run = run_metadata(run_id, config, options["source"])
    run.update(command="scan", input=str(input_path), options=options, options_sha256=json_sha256(options),
               manual_corners_used=False, manual_fallback_used=False, status="error")
    times, outputs = run["elapsed_ms"], run["outputs"]
    start = perf_counter()

    def save(key: str, filename: str, array) -> None:
        write_png(path / filename, array)
        outputs[key] = filename

    try:
        run["source"] = identified_source(input_path, options["source"])
        tick = perf_counter()
        photo = load_photo(input_path)
        times["decode"] = elapsed_ms(tick)
        run.update(input_sha256=photo.sha256, input_size=photo.size, raw_size=photo.raw_size,
                   exif_orientation=photo.exif_orientation)
        save("A", VARIANT_FILES["A"], photo.bgr)
        method = options["method"]
        selected = photo.bgr
        space = NORMALIZED_SPACE
        if method == "A":
            run["geometry"] = {"status": "not_applicable", "reason": "Raw-photo baseline A"}
        else:
            geometry = {"auto_status": "not_attempted"}
            run["geometry"] = geometry
            tick = perf_counter()
            points = None
            label = "manual"
            try:
                if corners_path is None or options["auto_first"]:
                    detection = detect_page(photo.bgr, config["geometry"], options["page_orientation"])
                    geometry.update(auto_status="candidate_found" if detection.corners is not None else "detection_failed",
                                    detection=detection.diagnostics)
                    points = detection.corners
                    label = "auto"
                if points is None and corners_path is not None:
                    geometry["manual_annotation"] = str(corners_path)
                    annotation = load_annotation(corners_path, photo)
                    if annotation.source and run["source"] not in ("unspecified", annotation.source):
                        raise DocscanError("source_mismatch", f"Corner annotation source is {annotation.source}")
                    if annotation.source:
                        run["source"] = annotation.source
                    run["manual_annotation_sha256"] = file_sha256(corners_path)
                    run["manual_corners_used"] = True
                    run["manual_fallback_used"] = label == "auto"
                    label = "manual_fallback" if label == "auto" else "manual"
                    points = annotation.corners
                    geometry["annotation_coordinate_space"] = annotation.input_coordinate_space
                if points is None:
                    raise DocscanError("detection_failed", geometry.get("detection", {}).get("reason", "No corners available"))
            except DocscanError as error:
                geometry.update(status="error", **error_record(error))
                save("detected_overlay", "detected_overlay.png", corner_overlay(photo.bgr, None, error.code))
                raise
            finally:
                times["locate"] = elapsed_ms(tick)
            save("detected_overlay", "detected_overlay.png", corner_overlay(photo.bgr, points, label))
            tick = perf_counter()
            warped, metadata = warp_document(photo.bgr, points, config["geometry"], options["page_orientation"])
            gray = gray_image(warped)
            times["warp_and_gray"] = elapsed_ms(tick)
            geometry.update(status="ok", corner_origin=label, **metadata)
            write_json(path / "geometry.json", geometry)
            outputs["geometry"] = "geometry.json"
            save("warped_color", "warped_color.png", warped)
            save("B", VARIANT_FILES["B"], gray)
            variants, background, branch_times, settings = make_variants(gray, config["enhance"])
            run["enhancement"] = settings
            times["variant_incremental"] = branch_times
            save("illumination_map", "illumination_map.png", background)
            for name in "CDEF":
                save(name, VARIANT_FILES[name], variants[name])
            selected = variants[method]
            space = WARPED_SPACE
        run["selected_image"] = outputs[method]
        run["coordinate_space"] = space
        ocr = None
        if engine is not None:
            from .ocr import save_ocr

            tick = perf_counter()
            ocr = engine.recognize(selected, space)
            times["ocr_call_including_first_load"] = elapsed_ms(tick)
            times["ocr_predict"] = ocr["elapsed_ms"]
            times["ocr_input_resize"] = ocr["input_resize_ms"]
            run.update(ocr_status=ocr["status"], ocr=engine.metadata())
            outputs.update(save_ocr(path, ocr))
            if ocr["status"] == "error":
                raise DocscanError(ocr["error_code"], ocr["error"])
        else:
            run["ocr_status"] = "not_requested"
        if options["pdf_mode"] != "none":
            from .export_pdf import export_pdf

            tick = perf_counter()
            run["pdf"] = export_pdf(path / "scan.pdf", selected,
                                    ocr if options["pdf_mode"] == "searchable" else None, config["pdf"])
            times["pdf"] = elapsed_ms(tick)
            outputs["pdf"] = "scan.pdf"
        run["status"] = "ok"
    except Exception as error:
        run.update(**error_record(error))
        if options["pdf_mode"] != "none" and "pdf" not in run:
            run["pdf"] = {"mode": options["pdf_mode"], "searchable": False, "status": "not_exported"}
        if run.get("geometry", {}).get("status") == "error":
            write_json(path / "geometry.json", run["geometry"])
            outputs["geometry"] = "geometry.json"
    finally:
        times["total_wall"] = elapsed_ms(start)
        finish_run(path, run)
    return run


def scan(input_path: Path, corners: Path | None, output: Path, config: dict, options: dict) -> dict:
    if options["pdf_mode"] == "searchable" and options["ocr"] == "none":
        raise DocscanError("pdf_requires_ocr", "--pdf-mode searchable requires --ocr paddle")
    engine = None
    if options["ocr"] == "paddle":
        from .ocr import PaddleEngine

        engine = PaddleEngine(config["ocr"])
    batch = input_path.is_dir()
    if batch:
        inputs = sorted((p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                        key=lambda path: (path.name.casefold(), path.name))
        if not inputs:
            raise DocscanError("empty_input_directory", f"No JPG/PNG inputs in {input_path}")
        if corners is not None and not corners.is_dir():
            raise DocscanError("invalid_corners_directory", "Directory scans require --corners to be a sidecar directory")
    else:
        if corners is not None and corners.is_dir():
            raise DocscanError("invalid_corners", "Single-image scans require --corners to be a JSON file")
        inputs = [input_path]
    path, run_id = new_run(output)
    if not batch:
        run = scan_one(input_path, corners, path, run_id, config, options, engine)
        return {"run_dir": str(path), "run_id": run_id, "status": run["status"],
                "error_code": run.get("error_code"), "error": run.get("error")}
    batch_run = run_metadata(run_id, config, options["source"])
    batch_run.update(command="scan_batch", input=str(input_path), options=options,
                     options_sha256=json_sha256(options), images=[])
    start = perf_counter()
    for index, image in enumerate(inputs, start=1):
        child = path / f"{index:03}_{image.stem[:80]}"
        child.mkdir()
        sidecar = corners / f"{image.stem}.corners.json" if corners is not None else None
        run = scan_one(image, sidecar, child, f"{run_id}/{index:03}", config, options, engine)
        batch_run["images"].append({"input": image.name, "source": run["source"], "run_dir": child.name,
                                    "input_sha256": run.get("input_sha256"), "status": run["status"],
                                    "error_code": run.get("error_code"), "manual_corners_used": run["manual_corners_used"],
                                    "manual_fallback_used": run["manual_fallback_used"]})
    sources = {image["source"] for image in batch_run["images"]}
    batch_run["source"] = next(iter(sources)) if len(sources) == 1 else "mixed"
    passed = sum(item["status"] == "ok" for item in batch_run["images"])
    batch_run.update(status="ok" if passed == len(inputs) else "incomplete", passed=passed, failed=len(inputs) - passed)
    batch_run["input_sha256"] = json_sha256([(row["input"], row["input_sha256"]) for row in batch_run["images"]])
    batch_run["outputs"]["images"] = [f"{item['run_dir']}/run.json" for item in batch_run["images"]]
    if engine is not None:
        batch_run["ocr"] = engine.metadata()
    batch_run["elapsed_ms"]["total_wall"] = elapsed_ms(start)
    finish_run(path, batch_run)
    return {"run_dir": str(path), "run_id": run_id, "status": batch_run["status"],
            "passed": passed, "failed": len(inputs) - passed}
