"""The scan and save operations called by the desktop window."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .candidates import CandidateScanner, PageAnalysis
from .common import DocscanError, write_json
from .enhance import gray_image
from .export_pdf import PdfSource, export_book_pdf
from .geometry import validate_quad
from .image_io import load_photo, write_png

DPI = 300
Progress = Callable[[str, int, int], None]
Point = tuple[float, float]


class ScanCancelled(Exception):
    pass


class PhotoProblem(DocscanError):
    def __init__(self, error: DocscanError, photo_index: int):
        super().__init__(error.code, str(error))
        self.photo_index = photo_index


@dataclass(frozen=True)
class PhotoBoundary:
    analysis: PageAnalysis
    size: tuple[int, int]
    sha256: str

    @property
    def points(self) -> tuple[Point, ...]:
        normalized = self.analysis.geometry.corners / (np.asarray(self.size) - 1)
        return tuple((float(x), float(y)) for x, y in np.clip(normalized, 0, 1))


@dataclass(frozen=True)
class PhotoInput:
    path: Path
    turns: int = 0
    corners: tuple[Point, ...] | None = None
    boundary: PhotoBoundary | None = None

    def turned(self) -> PhotoInput:
        corners = None
        if self.corners is not None:
            rotated = tuple((1 - y, x) for x, y in self.corners)
            corners = rotated[-1:] + rotated[:-1]
        return replace(self, turns=(self.turns + 1) % 4, corners=corners, boundary=None)


@dataclass(frozen=True)
class PagePreview:
    source: PdfSource
    variants: tuple[Path, ...]
    warning: str = ""


@dataclass(frozen=True)
class ScanPreview:
    pages: tuple[PagePreview, ...]
    photos: tuple[PhotoInput, ...]


def _checkpoint(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise ScanCancelled


def _may_be_blurred(image: np.ndarray) -> bool:
    # Judge printed detail on the prepared page. Skip blank and very sparse pages.
    gray = gray_image(image) if image.ndim == 3 else image
    height, width = gray.shape
    scale = min(1, 1100 / max(height, width))
    small = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    ink = small < 200
    if np.count_nonzero(ink) < 0.004 * small.size:
        return False
    edges = cv2.Laplacian(small, cv2.CV_32F)
    return float(np.mean(np.square(edges[ink]))) < 160


def _warning(image: np.ndarray, needs_review: bool) -> str:
    messages = []
    if needs_review:
        messages.append("未能可靠识别纸边，请拖动原图上的四个圆点。")
    if _may_be_blurred(image):
        messages.append("这一页疑似模糊，建议放大核对小字或重新拍摄。")
    return " ".join(messages)


def _inspect_photo(item: PhotoInput, index: int, scanner: CandidateScanner) -> tuple[PhotoInput, np.ndarray]:
    try:
        photo = load_photo(item.path)
        image = np.ascontiguousarray(np.rot90(photo.bgr, -item.turns))
        if item.boundary is not None:
            if item.boundary.sha256 != photo.sha256:
                raise DocscanError("photo_changed", "照片文件已变化。请移除后重新添加，再核对边界。")
            return item, image
        boundary = PhotoBoundary(scanner.analyze(image), image.shape[1::-1], photo.sha256)
        corners = item.corners
        if corners is not None:
            # After rotation, align the edited corner order with the newly detected page axes.
            points = np.asarray(corners)
            first = min(range(4), key=lambda i: np.square(np.roll(points, -i, axis=0) - boundary.points).sum())
            corners = corners[first:] + corners[:first]
        return replace(item, boundary=boundary, corners=corners), image
    except (DocscanError, OSError, RuntimeError, ValueError, cv2.error) as error:
        problem = error if isinstance(error, DocscanError) else DocscanError("scan_failed", str(error))
        raise PhotoProblem(problem, index) from error


def inspect_boundaries(items: Sequence[PhotoInput], config: dict,
                       progress: Progress, cancel: threading.Event) -> tuple[PhotoInput, ...]:
    scanner = CandidateScanner(config)
    inspected = []
    for index, item in enumerate(items):
        _checkpoint(cancel)
        progress("识别纸页边界", index + 1, len(items))
        detected, _ = _inspect_photo(item, index, scanner)
        inspected.append(detected)
    _checkpoint(cancel)
    return tuple(inspected)


def prepare_photos(items: Sequence[PhotoInput], directory: Path, config: dict,
                   progress: Progress, cancel: threading.Event) -> ScanPreview:
    if not items:
        raise DocscanError("empty_book", "请选择要处理的照片。")

    scanner = CandidateScanner(config)
    pages, inspected = [], []
    for index, item in enumerate(items):
        _checkpoint(cancel)
        progress("生成三种去影效果", index + 1, len(items))
        item, image = _inspect_photo(item, index, scanner)
        try:
            size = item.boundary.size
            manual = (validate_quad(np.asarray(item.corners) * (np.asarray(size) - 1), size,
                                    require_image_order=False) if item.corners else None)
            variants, info = scanner.scan(image, manual=manual, analysis=item.boundary.analysis)
            _checkpoint(cancel)
            folder = directory / f"{index + 1:04d}"
            folder.mkdir(parents=True)
            paths = tuple(folder / f"{label}.png" for label in "ABC")
            for label, path in zip("ABC", paths, strict=True):
                write_png(path, variants[label])
            page_height, page_width = variants["A"].shape
            source = PdfSource(paths[0], None, (page_width * 72 / DPI, page_height * 72 / DPI))
            review = info["mode"] == "full_frame_review" and not info["manual_boundary"]
            pages.append(PagePreview(source, paths, _warning(variants["A"], review)))
            inspected.append(item)
        except (DocscanError, OSError, RuntimeError, ValueError, cv2.error) as error:
            problem = error if isinstance(error, DocscanError) else DocscanError("scan_failed", str(error))
            raise PhotoProblem(problem, index) from error
    return ScanPreview(tuple(pages), tuple(inspected))


def save_scans(preview: ScanPreview, destination: Path, searchable: bool, config: dict,
               progress: Progress, cancel: threading.Event) -> dict:
    with TemporaryDirectory(prefix=".docscan-", dir=destination.parent) as folder:
        directory = Path(folder)
        sources = [page.source for page in preview.pages]
        if searchable:
            from .ocr import PaddleEngine

            engine = PaddleEngine(config["ocr"])
            for i, source in enumerate(sources, 1):
                _checkpoint(cancel)
                progress("识别文字", i, len(sources))
                result = engine.recognize(load_photo(source.image).bgr, "book_page_pixels")
                _checkpoint(cancel)
                if result["status"] == "error":
                    raise DocscanError(result["error_code"], result["error"])
                text = directory / f"{i:04d}.json"
                write_json(text, {key: value for key, value in result.items() if key != "raw_result"})
                sources[i - 1] = replace(source, ocr=text)

        def rendering(page: int, total: int) -> None:
            _checkpoint(cancel)
            progress("生成 PDF", page, total)

        pending = directory / "book.pdf"
        report = export_book_pdf(pending, sources, {**config["pdf"], "dpi": DPI}, on_page=rendering)
        _checkpoint(cancel)
        pending.replace(destination)
        return report
