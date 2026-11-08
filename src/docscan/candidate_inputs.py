from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
import pymupdf
from PIL import Image

from .common import DocscanError, error_record
from .image_io import IMAGE_SUFFIXES, load_photo


@dataclass(frozen=True)
class CandidateInput:
    folder: Path
    source: dict
    pixels: np.ndarray | None


def _natural_key(name: str) -> list[str | int]:
    return [int(part) if part.isascii() and part.isdecimal() else part
            for part in re.split(r"([0-9]+)", name.casefold())]


def _photo_on_page(page: pymupdf.Page, max_edge: int) -> tuple[np.ndarray, str, str]:
    images = page.get_image_info(xrefs=True)
    if (len(images) == 1 and images[0]["xref"] and page.rotation == 0 and not page.get_text().strip()
            and not page.get_drawings() and page.first_annot is None and page.first_widget is None):
        info = images[0]
        a, b, c, d, _, _ = info["transform"]
        if a > 0 and d > 0 and abs(b) + abs(c) < 1e-6 and page.rect.contains(pymupdf.Rect(info["bbox"])):
            original = page.parent.extract_image(info["xref"])
            if original and not original["smask"]:
                raw = original["image"]
                with Image.open(BytesIO(raw)) as image:
                    pixels = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                return pixels, "embedded_image", hashlib.sha256(raw).hexdigest()
    scale = max_edge / max(page.rect.width, page.rect.height)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
    raw = pixmap.samples
    rgb = np.frombuffer(raw, np.uint8).reshape(pixmap.height, pixmap.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "rendered_page", hashlib.sha256(raw).hexdigest()


def _pdf_photos(pdf: pymupdf.Document, name: str, group: int, max_edge: int) -> Iterator[CandidateInput]:
    prefix = Path(f"pdf-{group:02d}")
    if len(pdf) < 2:
        error = DocscanError("missing_photo_pages", f"{name} needs a reference page followed by at least one photo page")
        yield CandidateInput(prefix, {"name": name, **error_record(error)}, None)
        return
    for index in range(1, len(pdf)):
        page = index + 1
        source = {"name": name, "page": page}
        try:
            pixels, extraction, digest = _photo_on_page(pdf[index], max_edge)
            source.update(extraction=extraction, sha256=digest)
        except (OSError, ValueError, RuntimeError) as error:
            pixels = None
            source.update(**error_record(error))
        yield CandidateInput(prefix / f"page-{page:02d}", source, pixels)


def candidate_inputs(path: Path, max_edge: int) -> Iterator[CandidateInput]:
    if path.is_file() and path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                pdfs = sorted((entry for entry in archive.infolist() if not entry.is_dir()
                               and entry.filename.lower().endswith(".pdf")),
                              key=lambda item: _natural_key(item.filename))
                if not pdfs:
                    raise DocscanError("no_pdfs", f"No PDF groups found in {path}")
                for group, entry in enumerate(pdfs, 1):
                    try:
                        with pymupdf.open(stream=archive.read(entry), filetype="pdf") as pdf:
                            yield from _pdf_photos(pdf, entry.filename, group, max_edge)
                    except (OSError, ValueError, RuntimeError) as error:
                        yield CandidateInput(Path(f"pdf-{group:02d}"),
                                             {"name": entry.filename, **error_record(error)}, None)
        except BadZipFile as error:
            raise DocscanError("invalid_photo_archive", f"Cannot read {path}: {error}") from error
    elif path.is_file() and path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as pdf:
            yield from _pdf_photos(pdf, path.name, 1, max_edge)
    else:
        files = (sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                        key=lambda p: _natural_key(p.name)) if path.is_dir() else [path])
        if not files:
            raise DocscanError("no_photos", f"No JPG or PNG files found in {path}")
        for index, file in enumerate(files, 1):
            source = {"name": file.name}
            try:
                photo = load_photo(file)
                pixels = photo.bgr
                source.update(extraction="image_file", sha256=photo.sha256)
            except (OSError, DocscanError) as error:
                pixels = None
                source.update(**error_record(error))
            yield CandidateInput(Path(f"image-{index:04d}"), source, pixels)
