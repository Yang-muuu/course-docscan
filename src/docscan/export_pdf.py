from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf as pdf

from .common import DocscanError
from .image_io import encode_png, load_photo


def _background(document: pdf.Document, image: np.ndarray, width: float, height: float) -> pdf.Page:
    page = document.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=encode_png(image), keep_proportion=False)
    return page


def _line_frame(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = polygon[1] - polygon[0] + polygon[2] - polygon[3]
    length = np.linalg.norm(u)
    if length < 1e-6:
        raise DocscanError("invalid_pdf_polygon", "OCR line has no usable horizontal extent")
    u /= length
    v = np.array([-u[1], u[0]])
    xy = polygon @ np.column_stack((u, v))
    low, high = xy.min(axis=0), xy.max(axis=0)
    if min(high - low) < 1e-4:
        raise DocscanError("invalid_pdf_polygon", "OCR line has no usable area")
    return u, v, low, high


def _hidden_line(page: pdf.Page, line: dict, scale: np.ndarray, font: pdf.Font) -> None:
    text = line["text"]
    missing = sorted({character for character in text if not font.has_glyph(ord(character))})
    if missing:
        raise DocscanError("pdf_font_missing_glyph", f"PDF font lacks characters: {''.join(missing)!r}")
    u, v, low, high = _line_frame(np.array(line["polygon"], dtype=float) * scale)
    width, height = high - low
    fontsize = float(0.96 * height / (font.ascender - font.descender))
    stretch = width / font.text_length(text, fontsize=fontsize)
    base = low[0] * u + (low[1] + 0.02 * height + font.ascender * fontsize) * v
    origin = pdf.Point(*base)
    # TextWriter applies the morph in PDF's y-up space, while polygons use y-down pixels.
    morph = pdf.Matrix(float(stretch * u[0]), float(-stretch * u[1]), float(-v[0]), float(v[1]), 0, 0)
    writer = pdf.TextWriter(page.rect)
    writer.append(origin, text, font=font, fontsize=fontsize)
    writer.write_text(page, morph=(origin, morph), render_mode=3)


def _strip_space(text: str) -> str:
    return "".join(text.split())


def _default_keyword(text: str) -> str:
    chinese = re.search(r"[\u3400-\u9fff]{2,}", text)
    if chinese:
        return chinese.group()[:6]
    word = re.search(r"\S+", text)
    return word.group()[:12] if word else ""


def _verify_text(page: pdf.Page, expected: str, keyword: str | None, ocr: dict | None) -> dict:
    extracted = page.get_text("text", sort=False)
    if not _strip_space(expected) or _strip_space(extracted) != _strip_space(expected):
        raise DocscanError("pdf_text_mismatch", "PDF extraction differs from the OCR text or has no text")
    keyword = keyword if keyword is not None else _default_keyword(expected)
    hits = page.search_for(keyword, quads=True) if keyword.strip() else []
    if not hits:
        raise DocscanError("pdf_search_failed", f"Keyword not found in text layer: {keyword!r}")
    spans = sorted(page.get_texttrace(), key=lambda span: span["seqno"])
    if not spans or any(span["type"] != 3 for span in spans):
        raise DocscanError("pdf_text_visible", "PDF text layer is absent or not fully invisible")
    report = {"characters": len(_strip_space(extracted)), "keyword": keyword, "keyword_hits": len(hits),
              "text_matches": True, "invisible_text": True}
    if ocr is not None:
        w, h = ocr["image_size"]
        scale = np.array([page.rect.width / w, page.rect.height / h])
        chars = [(chr(char[0]), pdf.Rect(char[3])) for span in spans for char in span["chars"]
                 if not chr(char[0]).isspace()]
        pointer = 0
        line_coverages = []
        for line in ocr["lines"]:
            expected_line = _strip_space(line["text"])
            used = chars[pointer:pointer + len(expected_line)]
            pointer += len(expected_line)
            if "".join(char for char, _ in used) != expected_line:
                raise DocscanError("pdf_position_check_failed", "Cannot associate traced glyphs with OCR lines")
            u, v, low, high = _line_frame(np.array(line["polygon"], dtype=float) * scale)
            centers = np.array([[(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2] for _, box in used])
            coordinates = centers @ np.column_stack((u, v))
            tolerance = max(0.5, 0.12 * (high[1] - low[1]))
            inside = np.all((coordinates >= low - tolerance) & (coordinates <= high + tolerance), axis=1)
            line_coverages.append(float(inside.mean()))
        if pointer != len(chars) or not line_coverages or min(line_coverages) < 0.90:
            raise DocscanError("pdf_position_check_failed", "A text span is misplaced relative to its OCR polygon")
        report.update(position_checked=True, checked_lines=len(line_coverages), min_line_center_coverage=min(line_coverages))
    return report


def _encode_page(image: np.ndarray, ocr: dict | None, config: dict,
                 page_size: tuple[float, float] | None = None) -> tuple[bytes, dict]:
    h, w = image.shape[:2]
    factor = 72 / config["dpi"]
    width, height = page_size if page_size is not None else (w * factor, h * factor)
    if ocr is not None and (ocr["status"] != "ok" or not ocr["lines"]):
        raise DocscanError("no_searchable_text", "Searchable PDF requires a successful OCR result with text")
    if ocr is not None and ocr["image_size"] != [w, h]:
        raise DocscanError("pdf_coordinate_mismatch", "OCR image size differs from PDF background size")
    try:
        with pdf.open() as document:
            page = _background(document, image, width, height)
            if ocr is not None:
                font = pdf.Font(fontfile=config["font"]) if config["font"] else pdf.Font(fontname="cjk")
                for line in ocr["lines"]:
                    _hidden_line(page, line, np.array([page.rect.width / w, page.rect.height / h]), font)
                document.subset_fonts()
            encoded = document.tobytes(garbage=4, deflate=True)
        report = {"mode": "image", "searchable": False, "image_size": [w, h], "dpi": config["dpi"],
                  "page_size_points": [width, height],
                  "effective_dpi": [round(w * 72 / width, 3), round(h * 72 / height, 3)]}
        if ocr is not None:
            with pdf.open(stream=encoded, filetype="pdf") as check:
                verified = _verify_text(check[0], ocr["full_text"], None, ocr)
            report.update(mode="searchable", searchable=True, font=font.name, **verified)
        return encoded, report
    except (RuntimeError, ValueError, OSError) as error:
        raise DocscanError("pdf_export_failed", str(error)) from error


def export_pdf(path: Path, image: np.ndarray, ocr: dict | None, config: dict) -> dict:
    encoded, report = _encode_page(image, ocr, config)
    try:
        path.write_bytes(encoded)
    except OSError as error:
        raise DocscanError("pdf_export_failed", str(error)) from error
    return report


@dataclass(frozen=True)
class PdfSource:
    image: Path
    ocr: Path | None
    page_size: tuple[float, float] | None


def _verify_book(document: pdf.Document, expected: list[str], keyword: str | None = None) -> dict:
    if len(document) != len(expected):
        raise DocscanError("pdf_page_count", "PDF page count differs from the expected text pages")
    checks = []
    for number, (page, text) in enumerate(zip(document, expected, strict=True), 1):
        if _strip_space(text):
            check = _verify_text(page, text, None, None)
            checks.append({"page": number, "searchable": True, **check})
        else:
            if _strip_space(page.get_text()):
                raise DocscanError("pdf_text_mismatch", f"Unexpected text on PDF page {number}")
            checks.append({"page": number, "searchable": False, "characters": 0})
    report = {"pages": len(document), "searchable_pages": sum(page["searchable"] for page in checks), "page_checks": checks}
    if keyword is not None:
        hits = sum(len(page.search_for(keyword)) for page in document) if keyword.strip() else 0
        if not hits:
            raise DocscanError("pdf_search_failed", f"Keyword not found in the book: {keyword!r}")
        report.update(keyword=keyword, keyword_hits=hits)
    return report


def export_book_pdf(path: Path, sources: list[PdfSource], config: dict) -> dict:
    if not sources:
        raise DocscanError("empty_book", "There are no pages to export")
    staging = path.with_suffix(".partial.pdf")
    texts, reports = [], []
    try:
        with pdf.open() as document:
            for number, source in enumerate(sources, 1):
                result = json.loads(source.ocr.read_text(encoding="utf-8")) if source.ocr is not None else None
                layered = result if result is not None and result["status"] != "no_text" else None
                try:
                    encoded, report = _encode_page(load_photo(source.image).bgr, layered, config, source.page_size)
                except DocscanError as error:
                    raise DocscanError(error.code, f"PDF page {number}: {error}") from error
                with pdf.open(stream=encoded, filetype="pdf") as single:
                    document.insert_pdf(single)
                texts.append(layered["full_text"] if layered is not None else "")
                reports.append({"page": number, **report})
            if any(source.ocr is not None for source in sources) and not any(texts):
                raise DocscanError("no_searchable_text", "OCR did not find text on any page. Use image mode for a photo-only PDF.")
            document.save(staging, garbage=3, deflate=True)
        with pdf.open(staging) as check:
            verified = _verify_book(check, texts)
        staging.replace(path)
        return {**verified, "searchable": bool(verified["searchable_pages"]), "page_exports": reports}
    except (RuntimeError, ValueError, OSError) as error:
        raise DocscanError("pdf_export_failed", str(error)) from error
    finally:
        staging.unlink(missing_ok=True)


def verify_pdf(path: Path, expected: str, keyword: str | None = None, ocr: dict | None = None,
               background: Path | None = None, render: Path | None = None, *, page_number: int | None = None) -> dict:
    with pdf.open(path) as document:
        if page_number is None and len(document) > 1:
            if ocr is not None or background is not None or render is not None:
                raise DocscanError("pdf_page_required", "Use --page when checking a single background or OCR result in a book")
            report = _verify_book(document, expected.split("\f"), keyword)
            if not report["searchable_pages"]:
                raise DocscanError("no_searchable_text", "No PDF pages contain searchable text")
            return report
        index = 0 if page_number is None else page_number - 1
        if not 0 <= index < len(document):
            raise DocscanError("pdf_page_count", f"Page must be in the range 1–{len(document)}")
        if "\f" in expected:
            pages = expected.split("\f")
            if len(pages) != len(document):
                raise DocscanError("pdf_page_count", "PDF page count differs from the expected text pages")
            expected = pages[index]
        page = document[index]
        report = _verify_text(page, expected, keyword, ocr)
        if background is not None:
            image = load_photo(background).bgr
            h, w = image.shape[:2]
            if ocr is not None and ocr["image_size"] != [w, h]:
                raise DocscanError("pdf_coordinate_mismatch", "Background does not match OCR dimensions")
            zoom = min(2, 1600 / max(page.rect.width, page.rect.height))
            matrix = pdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            with pdf.open() as reference:
                reference_page = _background(reference, image, page.rect.width, page.rect.height)
                reference_pix = reference_page.get_pixmap(matrix=matrix)
                if pix.samples != reference_pix.samples:
                    raise DocscanError("pdf_background_mismatch", "Render differs from the scan-only background")
            report["render_equals_image_only"] = True
            if render is not None:
                render.parent.mkdir(parents=True, exist_ok=True)
                pix.save(render)
        elif render is not None:
            render.parent.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(dpi=150).save(render)
    return report
