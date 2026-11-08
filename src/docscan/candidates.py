"""Three printable scans of each independently processed photograph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from .candidate_geometry import PageGeometry, locate_document
from .candidate_inputs import candidate_inputs
from .common import DocscanError, elapsed_ms, error_record, finish_run, new_run, run_metadata
from .dewarp import Dewarper
from .enhance import gray_image
from .geometry import projected_page_aspect, rectified_size
from .image_io import write_png
from .models import CANDIDATE_MODELS, configure_paddle_cache, invalid_model_paths, model_status, paddle_model_path


PROFILES = {"A": "balanced", "B": "strong_shadow_removal", "C": "fine_detail"}


def _rectify(image: np.ndarray, page: PageGeometry, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    corners = page.corners
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    width, height = lengths[::2].mean(), lengths[1::2].mean()
    max_edge = settings["max_long_edge"]
    if page.mode in {"sheet", "book_leaf"}:
        # The source estimate can be stretched by perspective or a curved gutter.
        aspect = 0.665 if page.spine_side is not None else 210 / 297
        height = min(max_edge, max(3, round(height * 1.4)))
        width = max(3, round(height * aspect))
    else:
        geometry = {**settings, "page_format": "native"}
        aspect = projected_page_aspect(corners, (image.shape[1], image.shape[0]))
        width, height = rectified_size(corners, geometry, "portrait", aspect=aspect)
    if min(width, height) < 3:
        raise DocscanError("page_too_small", "The visible page is too small to rectify")
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(corners, target)
    pixels = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return pixels, transform


def _text_mask(polygons: np.ndarray, transform: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    if len(polygons):
        for quad in cv2.perspectiveTransform(polygons, transform):
            x, y = quad.mean(axis=0)
            if 0 <= x < width and 0 <= y < height:
                points = np.rint(np.clip(quad, [-width, -height], [2 * width, 2 * height])).astype(np.int32)
                cv2.fillConvexPoly(mask, points, 1)
    padding = max(3, round(0.012 * min(shape)) | 1)
    return cv2.dilate(mask, np.ones((padding, padding), np.uint8)) != 0


def _detail_ink(smooth: np.ndarray, background: np.ndarray, ink: np.ndarray,
                text: np.ndarray, window: int) -> np.ndarray:
    short = min(ink.shape)
    size = max(7, round(0.16 * short) | 1)
    upper = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    upper = cv2.medianBlur(upper, window).astype(np.float32)
    tones = np.clip((upper - background) / np.maximum(upper, 35), 0, 0.15)
    inset = max(1, round(0.08 * short))
    spread = np.ptp(np.quantile(background[inset:-inset, inset:-inset], [0.12, 0.88]))
    # A wider illumination window retains gray drawing fills. Blend less across a strong cast shadow.
    blend = 0.72 - 0.48 * np.clip((spread - 30) / 100, 0, 1)
    # A sharp change in the fine paper estimate usually indicates a shadow boundary.
    # Keep the wider drawing enhancement from bringing that boundary back.
    transition_size = max(5, round(0.043 * short) | 1)
    transition = cv2.morphologyEx(background, cv2.MORPH_GRADIENT,
                                 np.ones((transition_size, transition_size), np.uint8))
    blend *= 1 - 0.85 * np.clip((transition - 13) / 39, 0, 1)
    nearby = cv2.dilate(ink, np.ones((5, max(7, round(0.017 * short) | 1)), np.uint8))
    gain = np.clip(0.205 / np.maximum(nearby, 0.04), 1, 2.4)
    support = np.clip((nearby - 0.029) / 0.038, 0, 1)
    weak_strokes = ink * (gain - 1) * support * text
    span = max(7, round(0.055 * short) | 1)
    linework = cv2.dilate(ink, np.ones((span, span), np.uint8))
    linework = np.clip((linework - 0.06) / 0.10, 0, 1)
    text_weight = cv2.GaussianBlur(text.astype(np.float32), (0, 0), max(1, 0.009 * short))
    wider_tone = (blend * linework * (1 - text_weight) + 0.06 * text_weight) * tones
    return np.clip(ink + 0.72 * weak_strokes + wider_tone, 0, 1)


def _ink_strength(image: np.ndarray, binding: int | None, text: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = gray_image(image)
    height, width = gray.shape
    short = min(gray.shape)
    window = max(7, round(0.014 * short) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    # Closing removes thin ink from the illumination estimate. A local median follows cast shadows.
    smooth = cv2.medianBlur(gray, 3)
    background = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
    smoothing = max(5, round(0.022 * short) | 1)
    background = cv2.medianBlur(background, smoothing).astype(np.float32)
    level = np.maximum(background, 33)
    ink = np.clip((background - gray - 1.4) / level, 0, 1)
    # The fine estimate follows a hard shadow even where the wider window borrows light
    # from the other side. Closing on the median guide keeps narrow strokes out of it.
    radius = max(5, round(0.009 * short) | 1)
    guide = cv2.medianBlur(smooth, radius)
    fine = cv2.morphologyEx(guide, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))).astype(np.float32)
    strong_level = np.minimum(background, fine + np.maximum(2, 0.018 * background))
    strong = np.clip((strong_level - gray - 1.4) / np.maximum(strong_level, 33), 0, 1)
    ink = 0.72 * ink + 0.28 * strong
    if text is None:
        text = np.zeros(gray.shape, bool)
    detail = _detail_ink(smooth, background, ink, text, smoothing) if binding is not None else ink.copy()

    # Remove desk color or a dark object only when it reaches the outside of the proposed page.
    y, x = np.ogrid[:height, :width]
    margin = (np.minimum(y, height - 1 - y) < 0.16 * height) | (np.minimum(x, width - 1 - x) < 0.07 * width)
    saturation = np.ptp(image, axis=2) / (image.max(axis=2).astype(np.float32) + 24)
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    central = saturation[:, width // 5:width - width // 5]
    paper_tint = np.quantile(central, 0.80, axis=1)[:, None]
    tinted_background = saturation > np.maximum(0.30, paper_tint + 0.085)
    # Use the estimated paper brightness so dark glyphs are never seeds for background removal.
    outside = np.uint8(margin & (tinted_background | (background < 31)))
    outside = cv2.morphologyEx(outside, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels = cv2.connectedComponents(outside)
    touches = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    debris = cv2.dilate(np.uint8(np.isin(labels, touches[touches != 0])), np.ones((5, 5), np.uint8)) != 0
    outside = debris
    if binding is not None:
        edge = slice(-max(4, round(0.034 * width)), None) if binding == 1 else slice(0, max(4, round(0.034 * width)))
        band = ink[:, edge]
        fold = cv2.morphologyEx(np.uint8(band > 0.17), cv2.MORPH_OPEN,
                               np.ones((max(7, round(0.18 * height)), 1), np.uint8))
        outside[:, edge] |= cv2.dilate(fold, np.ones((3, 3), np.uint8)) != 0
    border = max(1, round(0.014 * short))
    trim = (np.minimum(y, height - 1 - y) < border) | (np.minimum(x, width - 1 - x) < border)
    clear = (outside | trim) & ~text
    ink[clear] = strong[clear] = detail[clear] = 0
    return ink, strong, detail


def _render(ink: np.ndarray, white: float, black: float, gamma: float, sharpen: float) -> np.ndarray:
    detail = np.clip(ink - cv2.GaussianBlur(ink, (0, 0), 0.67), -0.14, 0.13)
    # Isolated faint marks are more likely to be paper grain or ink showing through the sheet.
    nearby = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    threshold = white + 0.025 * np.clip((0.16 - nearby) / 0.12, 0, 1)
    normalized = np.clip((ink + sharpen * detail - threshold) / (black - threshold), 0, 1)
    return np.uint8(np.rint(255 - 248 * normalized ** gamma))


@dataclass(frozen=True)
class PageAnalysis:
    geometry: PageGeometry
    polygons: np.ndarray


class CandidateScanner:
    def __init__(self, config: dict):
        self.geometry = config["geometry"]
        settings = config["ocr"]
        root = Path(settings["model_dir"]).resolve()
        status = model_status(root, CANDIDATE_MODELS)
        invalid = invalid_model_paths(status)
        if invalid:
            raise DocscanError("model_file_changed", f"Invalid page model files: {', '.join(invalid)}")
        missing = [item["name"] for item in status.values() if not item["ready"]]
        if missing:
            raise DocscanError("candidate_models_missing", f"Run docscan prepare-models --candidates. Missing: {', '.join(missing)}")
        configure_paddle_cache(root)
        from paddleocr import DocImgOrientationClassification, TextDetection

        common = {"device": settings["device"], "enable_mkldnn": False, "cpu_threads": settings["cpu_threads"]}
        detector, orienter = status["detection"], status["orientation"]
        self.detector = TextDetection(model_name=detector["name"], model_dir=paddle_model_path(detector["directory"]),
                                     limit_side_len=1152, limit_type="max", box_thresh=0.45, **common)
        self.orienter = DocImgOrientationClassification(model_name=orienter["name"],
                                                        model_dir=paddle_model_path(orienter["directory"]), **common)
        self.dewarper = Dewarper(settings)
        self.model_hashes = {role: item["file_sha256"] for role, item in status.items()}

    def _flatten(self, ink: np.ndarray) -> np.ndarray:
        neutral = np.uint8(np.rint(255 - 255 * np.minimum(2 * ink, 1)))
        # White room around the photograph keeps the last line and page number inside the warp.
        room = max(4, round(0.028 * min(ink.shape)))
        padded = cv2.copyMakeBorder(neutral, room, room, room, room, cv2.BORDER_CONSTANT, value=255)
        flattened, _ = self.dewarper.apply(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR))
        normalized = cv2.resize(gray_image(flattened), ink.shape[1::-1], interpolation=cv2.INTER_AREA)
        return (255 - normalized).astype(np.float32) / 510

    def _text_polygons(self, image: np.ndarray) -> np.ndarray:
        if min(image.shape[:2]) < 3:
            raise DocscanError("page_too_small", "A photograph must have at least three pixels on each side")
        detected, = self.detector.predict(image, batch_size=1)
        return np.float64(detected["dt_polys"]).reshape(-1, 4, 2)

    def analyze(self, image: np.ndarray) -> PageAnalysis:
        polygons = self._text_polygons(image)
        return PageAnalysis(locate_document(image, polygons, self.geometry), polygons)

    def scan(self, image: np.ndarray, *, manual: np.ndarray | None = None,
             analysis: PageAnalysis | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if analysis is None:
            analysis = (self.analyze(image) if manual is None else
                        PageAnalysis(PageGeometry(manual, 0.0, None, "manual_edges"), self._text_polygons(image)))
        page = analysis.geometry if manual is None else replace(analysis.geometry, corners=manual)
        raw, transform = _rectify(image, page, self.geometry)
        text = _text_mask(analysis.polygons, transform, raw.shape[:2])
        ink, strong, detail = _ink_strength(raw, page.spine_side, text)
        balanced = _render(ink, 0.022, 0.425, 0.78, 0.34)
        analysis = cv2.cvtColor(balanced, cv2.COLOR_GRAY2BGR)
        original_scores, levelled_scores = self.orienter.predict([raw, analysis], batch_size=2, topk=4)
        score = {}
        for result, weight in ((original_scores, 0.34), (levelled_scores, 0.66)):
            for label, confidence in zip(result["label_names"], result["scores"], strict=True):
                score[int(label)] = score.get(int(label), 0.0) + weight * float(confidence)
        directions = (0, 180) if page.mode in {"sheet", "book_leaf"} else (0, 90, 180, 270)
        angle = max(directions, key=score.__getitem__)
        ink = np.ascontiguousarray(np.rot90(ink, angle // 90))
        strong = np.ascontiguousarray(np.rot90(strong, angle // 90))
        detail = np.ascontiguousarray(np.rot90(detail, angle // 90))
        metadata = {"mode": page.mode, "source_corners": page.corners.tolist(), "spine_side": page.spine_side,
                    "manual_boundary": manual is not None,
                    "text_angle_degrees": page.text_angle, "upright_rotation_degrees": angle,
                    "orientation_confidence": score[angle], "flattened": False}
        metadata["corners_extend_past_frame"] = bool(np.any(page.corners < 0) or
                                                     np.any(page.corners > np.array(image.shape[1::-1]) - 1))
        if page.spine_side is not None:
            # UVDoc moves the recorded pixels to flatten the book, without synthesizing any marks.
            ink = self._flatten(ink)
            strong = self._flatten(strong)
            detail = self._flatten(detail)
            metadata["flattened"] = True
        return {
            "A": _render(ink, 0.022, 0.425, 0.78, 0.34),
            "B": _render(strong, 0.063, 0.285, 0.67, 0.55),
            "C": _render(detail, 0.017, 0.47, 0.85, 0.26),
        }, metadata


def scan_candidates(input_path: Path, output: Path, config: dict) -> dict:
    folder, run_id = new_run(output)
    run = run_metadata(run_id, config, "user_input")
    run.update(command="candidates", input=str(input_path.resolve()), profiles=PROFILES, photos=[])
    start = perf_counter()
    try:
        scanner = CandidateScanner(config)
        run["models"] = scanner.model_hashes
        for item in candidate_inputs(input_path, config["geometry"]["max_long_edge"]):
            record = {"folder": item.folder.as_posix(), "source": item.source, "status": "error"}
            run["photos"].append(record)
            if item.pixels is None:
                continue
            try:
                variants, geometry = scanner.scan(item.pixels)
                destination = folder / item.folder
                destination.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(prefix=".page-", dir=destination.parent) as staging:
                    staged = Path(staging) / "candidates"
                    staged.mkdir()
                    for label, pixels in variants.items():
                        write_png(staged / f"{label}.png", pixels)
                    staged.rename(destination)
                record.update(status="ok", geometry=geometry,
                              images={label: (item.folder / f"{label}.png").as_posix() for label in variants})
            except (DocscanError, OSError, RuntimeError, ValueError, cv2.error) as error:
                record.update(**error_record(error))
        run["status"] = "ok" if run["photos"] and all(row["status"] == "ok" for row in run["photos"]) else "error"
    except (DocscanError, OSError, RuntimeError, ValueError, ImportError) as error:
        run.update(status="error", **error_record(error))
    finally:
        run["elapsed_ms"]["total"] = elapsed_ms(start)
        finish_run(folder, run)
    finished = sum(row["status"] == "ok" for row in run["photos"])
    return {"run_dir": str(folder.resolve()), "status": run["status"], "photos": len(run["photos"]),
            "processed": finished, "images": len(PROFILES) * finished, "error": run.get("error")}
