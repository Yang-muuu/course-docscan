from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np

from .common import SEED, DocscanError, dependency_versions, elapsed_ms, error_record, write_json
from .models import MODEL_NAMES, configure_paddle_cache, invalid_model_paths, model_status


def _row_groups(positions: list[tuple]) -> list[list[tuple]]:
    rows = []
    for position in sorted(positions, key=lambda item: (item[0], item[1])):
        if rows and abs(position[0] - rows[-1][0][0]) <= 0.5 * min(position[2], rows[-1][0][2]):
            rows[-1].append(position)
        else:
            rows.append([position])
    return rows


def _column_groups(positions: list[tuple], split: float) -> list[list[tuple]]:
    spanning, body = [], []
    for position in positions:
        xs = [point[0] for point in position[3]["polygon"]]
        (spanning if min(xs) < split < max(xs) else body).append(position)
    headings = _row_groups(spanning)
    breaks = [row[0][0] for row in headings]
    sections = [([], []) for _ in range(len(headings) + 1)]
    for position in body:
        section = sections[bisect_right(breaks, position[0])]
        section[int(position[1] >= split)].append(position)
    rows = []
    for index, (left, right) in enumerate(sections):
        if index:
            rows.append(headings[index - 1])
        rows.extend(_row_groups(left))
        rows.extend(_row_groups(right))
    return rows


def order_lines(lines: list[dict], *, column_split: float | None = None) -> tuple[list[dict], str]:
    if not lines:
        return [], ""
    slopes = []
    for line in lines:
        a, b, c, d = line["polygon"]
        dx = (b[0] - a[0]) + (c[0] - d[0])
        dy = (b[1] - a[1]) + (c[1] - d[1])
        if dx > 0 and abs(dy / dx) <= 0.5:
            slopes.append(dy / dx)
    slope = median(slopes) if slopes else 0.0
    positions = []
    for line in lines:
        polygon = np.asarray(line["polygon"], dtype=float)
        center_x = polygon[:, 0].mean()
        ys = polygon[:, 1] - slope * polygon[:, 0]
        positions.append((float(ys.mean()), float(center_x), max(1.0, float(np.ptp(ys))), line))
    rows = _row_groups(positions) if column_split is None else _column_groups(positions, column_split)
    ordered, text_rows = [], []
    for row_index, row in enumerate(rows, start=1):
        row = sorted(row, key=lambda item: item[1])
        text_rows.append(" ".join(item[3]["text"] for item in row))
        for _, _, _, line in row:
            ordered.append({**line, "reading_order": len(ordered) + 1, "row": row_index})
    return ordered, "\n".join(text_rows)


def _paddle_model_path(directory: str) -> str:
    if os.name == "nt" and not directory.isascii():
        # Paddle's Windows file reader accepts ASCII paths. The launcher works
        # from the bundle, so "models/..." also works in a Chinese directory.
        path = Path(directory)
        current = Path.cwd().resolve()
        if path.is_relative_to(current):
            relative = str(path.relative_to(current))
            if relative.isascii():
                return relative
    return directory


class PaddleEngine:
    def __init__(self, settings: dict, *, reading_order: str = "rows"):
        self.settings = settings.copy()
        if reading_order not in {"rows", "two-column"}:
            raise DocscanError("invalid_reading_order", f"Unsupported reading order: {reading_order}")
        self.reading_order = reading_order
        self.root = Path(settings["model_dir"]).resolve()
        self.options = {
            "use_doc_orientation_classify": False, "use_doc_unwarping": False,
            "use_textline_orientation": False, "device": settings["device"],
            "enable_mkldnn": False, "enable_hpi": False, "cpu_threads": settings["cpu_threads"],
            "text_det_limit_side_len": settings["max_input_side"], "text_det_limit_type": "max",
            "text_det_thresh": 0.3, "text_det_box_thresh": 0.6, "text_det_unclip_ratio": 1.5,
            "text_rec_score_thresh": settings["score_thresh"], "text_recognition_batch_size": 6,
        }
        self.versions = dependency_versions()
        self.model_load_ms = None
        self.model_file_sha256 = None
        self._pipeline = None
        self._load_error = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        start = perf_counter()
        try:
            missing = [name for name in ("paddlepaddle", "paddleocr", "paddlex") if self.versions[name] is None]
            if missing:
                raise DocscanError("ocr_dependency_missing", f"Missing packages: {', '.join(missing)}. Install .[ocr].")
            status = model_status(self.root)
            invalid = invalid_model_paths(status)
            if invalid:
                raise DocscanError("model_file_changed", f"Invalid model files: {', '.join(invalid)}. Move the affected cache directory aside and run prepare-models.")
            missing_models = [entry["name"] for entry in status.values() if not entry["ready"]]
            if missing_models:
                raise DocscanError("ocr_model_missing", f"Missing local model(s): {', '.join(missing_models)}. Run prepare-models.")
            configure_paddle_cache(self.root)
            import paddle
            from paddleocr import PaddleOCR

            paddle.seed(SEED)
            self.model_file_sha256 = {role: model["file_sha256"] for role, model in status.items()}
            self._pipeline = PaddleOCR(
                text_detection_model_name=MODEL_NAMES["detection"],
                text_recognition_model_name=MODEL_NAMES["recognition"],
                text_detection_model_dir=_paddle_model_path(status["detection"]["directory"]),
                text_recognition_model_dir=_paddle_model_path(status["recognition"]["directory"]),
                **self.options,
            )
        except Exception as error:
            if isinstance(error, DocscanError):
                self._load_error = error
                raise
            self._load_error = DocscanError("ocr_model_load_failed", f"{type(error).__name__}: {error}")
            raise self._load_error from error
        finally:
            self.model_load_ms = elapsed_ms(start)

    def metadata(self) -> dict:
        return {
            "engine": "paddleocr", "engine_version": self.versions["paddleocr"],
            "paddle_version": self.versions["paddlepaddle"], "paddlex_version": self.versions["paddlex"],
            "model_names": MODEL_NAMES, "model_file_sha256": self.model_file_sha256,
            "device": self.settings["device"],
            "reading_order": self.reading_order,
            "effective_options": {**self.options, "external_max_input_side": self.settings["max_input_side"]},
            "model_load_ms": self.model_load_ms, "ocr_result_reused": False,
        }

    def recognize(self, image: np.ndarray, coordinate_space: str) -> dict:
        h, w = image.shape[:2]
        result = {"status": "error", "image_size": [w, h], "coordinate_space": coordinate_space,
                  "lines": [], "full_text": "", "elapsed_ms": None, "input_resize_ms": None,
                  "inference_attempted": False, "timing_scope": "predict only; excludes model loading and input resize"}
        start = None
        try:
            self.load()
            tick = perf_counter()
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            scale = min(1, self.settings["max_input_side"] / max(w, h))
            iw, ih = max(1, round(w * scale)), max(1, round(h * scale))
            if (iw, ih) != (w, h):
                image = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA)
            result["model_input_size"] = [iw, ih]
            result["input_resize_ms"] = elapsed_ms(tick)
            result["inference_attempted"] = True
            start = perf_counter()
            pages = self._pipeline.predict(image)
            result["elapsed_ms"] = elapsed_ms(start)
            if len(pages) != 1:
                raise DocscanError("ocr_invalid_result", f"Expected one page, got {len(pages)}")
            page = pages[0]
            polys, texts, scores = (page[key] for key in ("rec_polys", "rec_texts", "rec_scores"))
            if len(polys) != len(texts) or len(scores) != len(texts):
                raise DocscanError("ocr_invalid_result", "Polygon, text and score counts differ")
            lines = []
            for polygon, text, confidence in zip(polys, texts, scores, strict=True):
                if not text.strip():
                    continue
                polygon = np.asarray(polygon, dtype=np.float64)
                confidence = float(confidence)
                if polygon.shape != (4, 2) or not np.isfinite(polygon).all() or not math.isfinite(confidence):
                    raise DocscanError("ocr_invalid_result", "Invalid OCR quadrilateral or confidence")
                polygon = np.clip((polygon + 0.5) * [w / iw, h / ih] - 0.5, [0, 0], [w - 1, h - 1])
                lines.append({"polygon": polygon.tolist(), "text": text, "confidence": confidence})
            split = w / 2 if self.reading_order == "two-column" else None
            result["lines"], result["full_text"] = order_lines(lines, column_split=split)
            result["raw_result"] = page.json
            result["status"] = "ok" if result["lines"] else "no_text"
        except Exception as error:
            if start is not None and result["elapsed_ms"] is None:
                result["elapsed_ms"] = elapsed_ms(start)
            result.update(error_record(error if isinstance(error, DocscanError) else
                                       DocscanError("ocr_inference_failed", f"{type(error).__name__}: {error}")))
        result.update(self.metadata())
        return result


def save_ocr(folder: Path, result: dict) -> dict:
    outputs = {"ocr": "ocr.json"}
    if "raw_result" in result:
        write_json(folder / "ocr_raw.json", result["raw_result"])
        outputs["ocr_raw"] = "ocr_raw.json"
    write_json(folder / "ocr.json", {key: value for key, value in result.items() if key != "raw_result"})
    if result["status"] in {"ok", "no_text"}:
        text = result["full_text"]
        (folder / "text.txt").write_text(text + "\n" if text else "", encoding="utf-8")
        outputs["text"] = "text.txt"
    return outputs


def download_record(engine: PaddleEngine) -> dict | None:
    try:
        record = json.loads((engine.root / "model_preparation.json").read_text(encoding="utf-8"))
        if record.get("status") == "ok" and record.get("model_file_sha256") == engine.model_file_sha256:
            return record
    except (OSError, ValueError):
        pass
    return None
