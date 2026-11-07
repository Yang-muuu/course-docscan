"""The scan and save operations called by the desktop window."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .capture_fusion import CaptureMismatch, fuse_captures
from .common import DocscanError, write_json
from .enhance import gray_image
from .export_pdf import PdfSource, export_book_pdf
from .geometry import validate_quad
from .image_io import load_photo, write_png
from .page_surface import prepare_page
from .spread import find_gutter, split_spread

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
class PhotoInput:
    path: Path
    turns: int = 0
    corners: tuple[Point, ...] | None = None
    gutter: tuple[float, float] | None = None

    def turned(self) -> PhotoInput:
        corners = None
        if self.corners is not None:
            rotated = tuple((1 - y, x) for x, y in self.corners)
            corners = rotated[-1:] + rotated[:-1]
        return replace(self, turns=(self.turns + 1) % 4, corners=corners, gutter=None)


@dataclass(frozen=True)
class PagePreview:
    source: PdfSource
    photo_index: int
    side: str
    warning: str = ""


@dataclass(frozen=True)
class ScanPreview:
    pages: tuple[PagePreview, ...]
    hints: tuple[PhotoInput, ...]
    merged: bool = False


def _checkpoint(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise ScanCancelled


def _may_be_blurred(image: np.ndarray) -> bool:
    # Judge printed detail on the prepared page. Skip blank and very sparse pages.
    gray = gray_image(image)
    height, width = gray.shape
    scale = min(1, 1100 / max(height, width))
    small = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    ink = small < 200
    if np.count_nonzero(ink) < 0.004 * small.size:
        return False
    edges = cv2.Laplacian(small, cv2.CV_32F)
    return float(np.mean(np.square(edges[ink]))) < 160


def _warning(image: np.ndarray, info: dict) -> str:
    messages = []
    if info["needs_review"]:
        messages.append("请核对裁剪范围，可用「调整边界」重新框选。")
    if _may_be_blurred(image):
        messages.append("这一页疑似模糊，建议放大核对小字或重新拍摄。")
    return " ".join(messages)


def prepare_photos(items: Sequence[PhotoInput], mode: str, directory: Path, config: dict,
                   progress: Progress, cancel: threading.Event) -> ScanPreview:
    if not items:
        raise DocscanError("empty_book", "请选择要处理的照片。")
    if mode == "same" and not 2 <= len(items) <= 4:
        raise DocscanError("invalid_book_options", "同页多拍需要 2 到 4 张同一页的完整照片。")
    directory.mkdir(parents=True, exist_ok=True)
    pages, hints, captures = [], [], []
    for index, item in enumerate(items):
        _checkpoint(cancel)
        progress("处理照片", index + 1, len(items))
        try:
            upright = load_photo(item.path).bgr
            image = np.ascontiguousarray(np.rot90(upright, -item.turns))
            height, width = image.shape[:2]
            hint = item
            manual = None
            if mode == "spread":
                gutter = item.gutter or find_gutter(image)[0]
                if gutter is None:
                    raise DocscanError("gutter_not_found", "未能确认书脊位置。请调整分割线后重试。")
                hint = replace(item, gutter=gutter)
                parts = [(side, pixels) for side, pixels, _ in split_spread(image, gutter, config["geometry"])]
            else:
                parts = [("single", image)]
                if item.corners is not None:
                    manual = validate_quad(np.asarray(item.corners) * [width - 1, height - 1], (width, height))

            for side, pixels in parts:
                _checkpoint(cancel)
                page, mask, info = prepare_page(pixels, config, "native", "auto", manual, dpi=DPI)
                if side == "single" and "geometry" in info:
                    normalized = np.asarray(info["geometry"]["points"]) / [width - 1, height - 1]
                    hint = replace(hint, corners=tuple((float(x), float(y)) for x, y in normalized))
                if mode == "same":
                    if "geometry" not in info or any(info["frame_edges_touched"].values()):
                        raise DocscanError("incomplete_capture", "合成需要每张照片都拍到完整纸页。")
                    captures.append((page, mask, info))
                else:
                    path = directory / f"{len(pages) + 1:04d}.png"
                    write_png(path, page)
                    size = info["pdf_page_size_points"]
                    pages.append(PagePreview(PdfSource(path, None, size), index, side, _warning(page, info)))
            hints.append(hint)
        except (DocscanError, OSError, ValueError, cv2.error, np.linalg.LinAlgError) as error:
            problem = error if isinstance(error, DocscanError) else DocscanError("scan_failed", str(error))
            raise PhotoProblem(problem, index) from error

    if mode == "same":
        _checkpoint(cancel)
        progress("对齐并合成", len(items), len(items))
        try:
            page, anchor, _ = fuse_captures([capture[0] for capture in captures])
        except CaptureMismatch as error:
            raise PhotoProblem(error, int(error.evidence["capture"]) - 1) from error
        _, mask, info = captures[anchor]
        page[mask == 0] = 255
        path = directory / "merged.png"
        write_png(path, page)
        pages.append(PagePreview(PdfSource(path, None, info["pdf_page_size_points"]), anchor, "merged",
                                 _warning(page, info)))
    _checkpoint(cancel)
    return ScanPreview(tuple(pages), tuple(hints), mode == "same")


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
