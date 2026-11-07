from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .common import DocscanError, elapsed_ms, error_record, file_sha256, finish_run, json_sha256, new_run, run_metadata
from .dataset import identified_source
from .export_pdf import PdfSource, export_book_pdf
from .geometry import load_annotation
from .image_io import IMAGE_SUFFIXES, load_photo, write_png
from .page_surface import fit_a4_after_dewarp, pdf_raster_layout, prepare_page, resize_page
from .spread import find_gutter, split_spread

BOOK_SPACE = "book_page_pixels"


def _inputs(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]

    def page_order(image: Path):
        tokens = re.split(r"([0-9]+)", image.name.casefold())
        return tuple(int(token) if index % 2 else token for index, token in enumerate(tokens)), image.name

    images = sorted((item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
                    key=page_order)
    if not images:
        raise DocscanError("empty_input_directory", f"No JPG/PNG inputs in {path}")
    return images


def _recognize(engine, image: np.ndarray, folder: Path, entry: dict) -> tuple[Path | None, str]:
    if engine is None:
        return None, ""
    from .ocr import save_ocr

    result = engine.recognize(image, BOOK_SPACE)
    save_ocr(folder, result)
    entry["ocr_status"] = result["status"]
    if result["status"] == "error":
        raise DocscanError(result["error_code"], result["error"])
    return folder / "ocr.json", result["full_text"]


def _fuse_book_page(root: Path, run: dict, sources: list[PdfSource], engine) -> tuple[PdfSource, str]:
    from .capture_fusion import CaptureMismatch, fuse_captures

    captures = run["pages"]
    if any("geometry" not in page or any(page["frame_edges_touched"].values()) for page in captures):
        raise DocscanError("incomplete_capture", "Same-page fusion requires a complete sheet in every photo")
    try:
        merged, anchor, registration = fuse_captures([load_photo(source.image).bgr for source in sources])
    except CaptureMismatch as error:
        run["fusion"] = {"status": "needs_review", **error.evidence}
        for page in captures:
            page["needs_review"] = True
        raise
    parent = captures[anchor]
    mask = load_photo(root / parent["mask"]).bgr[:, :, 0]
    merged[mask == 0] = 255
    folder = root / "merged"
    folder.mkdir()
    path = folder / "page.png"
    write_png(path, merged)
    entry = {**parent, "page": 1, "mode": "same_page", "image": path.relative_to(root).as_posix()}
    ocr_path, text = _recognize(engine, merged, folder, entry)
    run["capture_pages"] = captures
    run["pages"] = [entry]
    for capture in run["images"]:
        capture["capture_pages"] = capture["pages"]
        capture["pages"] = [1]
    run["fusion"] = {"status": "ok", "anchor_capture": anchor + 1, "registration": registration}
    return PdfSource(path, ocr_path, sources[anchor].page_size), text


def convert_book(input_path: Path, output: Path, corners_path: Path | None, config: dict, options: dict) -> dict:
    inputs = _inputs(input_path)
    dpi = options["dpi"]
    same_page = options.get("same_page", False)
    if not 72 <= dpi <= 400:
        raise DocscanError("invalid_pdf_dpi", "--dpi must be between 72 and 400")
    if same_page and (not 2 <= len(inputs) <= 4 or options["layout"] != "single" or options["dewarp"] != "none"):
        raise DocscanError("invalid_book_options", "--same-page needs 2–4 photos in a directory, --layout single and --dewarp none")
    if corners_path is not None and options["layout"] != "single":
        raise DocscanError("invalid_book_options", "Use --corners with one-page photos and --gutter with spreads")
    if options["gutter"] is not None and options["layout"] != "spread":
        raise DocscanError("invalid_book_options", "--gutter requires --layout spread")
    if corners_path is not None and input_path.is_dir() != corners_path.is_dir():
        raise DocscanError("invalid_corners", "Use a corner JSON for one photo, or a sidecar directory for a batch")
    root, run_id = new_run(output)
    run = run_metadata(run_id, config, options["source"])
    run.update(command="to-pdf", input=str(input_path), options=options, options_sha256=json_sha256(options),
               images=[], pages=[], status="error")
    sources, texts = [], []
    engine, dewarper = None, None
    start = perf_counter()
    try:
        if options["pdf_mode"] == "searchable":
            from .ocr import PaddleEngine

            engine = PaddleEngine(config["ocr"], reading_order=options.get("reading_order", "rows"))
        if options["dewarp"] == "uvdoc":
            from .dewarp import Dewarper

            dewarper = Dewarper(config["ocr"])
        for image_path in inputs:
            capture = {"input": str(image_path), "source": options["source"], "pages": [], "status": "error"}
            run["images"].append(capture)
            try:
                photo = load_photo(image_path)
                capture.update(input_sha256=photo.sha256, size=photo.size,
                               source=identified_source(image_path, options["source"]),
                               exif_orientation=photo.exif_orientation)
                annotation = None
                if corners_path is not None:
                    sidecar = corners_path / f"{image_path.stem}.corners.json" if input_path.is_dir() else corners_path
                    annotation = load_annotation(sidecar, photo)
                    if annotation.source and capture["source"] not in {"unspecified", annotation.source}:
                        raise DocscanError("source_mismatch", f"Corner annotation source is {annotation.source}")
                    capture["source"] = annotation.source or capture["source"]
                    capture["corners_sha256"] = file_sha256(sidecar)
                parts = [("single", photo.bgr, {})]
                if options["layout"] == "spread":
                    gutter, evidence = (options["gutter"], {"method": "manual"}) if options["gutter"] else find_gutter(photo.bgr)
                    capture["gutter"] = {"top_bottom_fractions": gutter, **evidence}
                    if gutter is None:
                        parts = [("unsplit", photo.bgr, {})]
                    else:
                        parts = split_spread(photo.bgr, gutter, config["geometry"])
            except (OSError, ValueError, DocscanError, cv2.error, np.linalg.LinAlgError) as error:
                capture.update(**error_record(error))
                continue

            capture["status"] = "ok"
            for side, pixels, split in parts:
                number = len(run["pages"]) + 1
                capture["pages"].append(number)
                entry = {"page": number, "side": side, "input": str(image_path), "split_transform": split,
                         "status": "error", "source": capture["source"]}
                run["pages"].append(entry)
                try:
                    apply_dewarp = dewarper is not None and side != "unsplit"
                    page, mask, prep = prepare_page(pixels, config, options["page_format"], options["page_orientation"],
                                                    annotation.corners if annotation is not None else None,
                                                    dpi=None if apply_dewarp else dpi)
                    entry.update(**prep)
                    entry["needs_review"] = entry["needs_review"] or side == "unsplit"
                    entry["geometry_input_space"] = "split_pixels" if split else "exif_normalized_pixels"
                    folder = root / "pages" / f"{number:04d}"
                    folder.mkdir(parents=True)
                    write_png(folder / "paper_mask.png", mask)
                    if apply_dewarp:
                        write_png(folder / "before_dewarp.png", page)
                        page, entry["dewarp"] = dewarper.apply(page)
                        if options["page_format"] == "a4":
                            page = fit_a4_after_dewarp(page, prep["orientation"], config["geometry"]["max_long_edge"])
                        page_size, target = pdf_raster_layout((page.shape[1], page.shape[0]), options["page_format"],
                                                               prep["orientation"], config["pdf"]["dpi"], dpi)
                        page = resize_page(page, target)
                        entry["pdf_page_size_points"] = page_size
                        entry["mask_coordinate_space"] = "before_dewarp_pixels"
                    elif dewarper is not None:
                        entry["dewarp"] = {"status": "skipped_unsplit"}
                    page_path = folder / "page.png"
                    write_png(page_path, page)
                    entry.update(image=page_path.relative_to(root).as_posix(), size=[page.shape[1], page.shape[0]],
                                 mask=(folder / "paper_mask.png").relative_to(root).as_posix())
                    ocr_path, text = _recognize(None if same_page else engine, page, folder, entry)
                    sources.append(PdfSource(page_path, ocr_path, entry["pdf_page_size_points"]))
                    texts.append(text)
                    entry["status"] = "ok"
                except (OSError, ValueError, RuntimeError, DocscanError, cv2.error, np.linalg.LinAlgError) as error:
                    entry.update(**error_record(error))
                    capture["status"] = "error"

        if any(image["status"] != "ok" for image in run["images"]):
            run.update(status="incomplete", error_code="incomplete_book", error="At least one photo or page failed. See run.json.")
        else:
            if same_page:
                source, text = _fuse_book_page(root, run, sources, engine)
                sources, texts = [source], [text]
            run["pdf"] = export_book_pdf(root / "book.pdf", sources, {**config["pdf"], "dpi": dpi})
            run["outputs"]["pdf"] = "book.pdf"
            if engine is not None:
                (root / "book.txt").write_text("\f".join(texts) + "\n", encoding="utf-8")
                run["outputs"]["text"] = "book.txt"
            run["status"] = "ok"
    except (OSError, ValueError, RuntimeError, DocscanError, cv2.error, np.linalg.LinAlgError) as error:
        run.update(**error_record(error))
    finally:
        provenance = [(image["input"], image.get("input_sha256")) for image in run["images"]]
        run["input_sha256"] = json_sha256(provenance) if provenance else None
        labels = {image["source"] for image in run["images"]}
        if labels:
            run["source"] = labels.pop() if len(labels) == 1 else "mixed"
        run["review_pages"] = [page["page"] for page in run["pages"] if page.get("needs_review")]
        if engine is not None:
            run["ocr"] = engine.metadata()
        if dewarper is not None:
            run["dewarp"] = dewarper.metadata
        run["elapsed_ms"]["total_wall"] = elapsed_ms(start)
        finish_run(root, run)
    return {"run_dir": str(root), "run_id": run_id, "status": run["status"], "pdf": run["outputs"].get("pdf"),
            "pages": run.get("pdf", {}).get("pages", 0), "searchable_pages": run.get("pdf", {}).get("searchable_pages", 0),
            "review_pages": run["review_pages"], "error_code": run.get("error_code"), "error": run.get("error")}
