from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .common import DocscanError
from .image_io import NORMALIZED_SPACE, Photo, raw_to_upright

RAW_SPACE = "raw_image_pixels"
CORNER_ORDER = "tl_tr_br_bl"


@dataclass(frozen=True)
class Annotation:
    corners: np.ndarray
    text_polygons: tuple[np.ndarray, ...]
    source: str | None
    input_coordinate_space: str


def polygon_array(points: object) -> np.ndarray:
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise DocscanError("invalid_corners", "Expected four finite [x,y] coordinates")
    try:
        polygon = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise DocscanError("invalid_corners", "Expected four finite [x,y] coordinates") from error
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise DocscanError("invalid_corners", "Expected four finite [x,y] coordinates")
    return polygon


def signed_area(points: np.ndarray) -> float:
    x, y = points.T
    return float((x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2)


def validate_quad(points: object, size: tuple[int, int], *, require_image_order: bool = True) -> np.ndarray:
    quad = polygon_array(points)
    if np.any(quad < 0) or np.any(quad > np.asarray(size) - 1):
        raise DocscanError("invalid_corners", f"Corner outside image bounds {size}")
    edges = np.roll(quad, -1, axis=0) - quad
    if np.min(np.linalg.norm(edges, axis=1)) < 2:
        raise DocscanError("invalid_corners", "Repeated or near-identical corners")
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    area = signed_area(quad)
    if area < max(4, 0.0001 * size[0] * size[1]) or np.min(turns) <= max(0.25, area * 0.0001):
        raise DocscanError("invalid_corners", "Corners must form a clockwise convex, non-degenerate quadrilateral")
    rightward = edges[0] - edges[2]
    downward = edges[1] - edges[3]
    if require_image_order and (rightward[0] <= 0 or downward[1] <= 0):
        raise DocscanError("invalid_corners", "Corner order must be top-left, top-right, bottom-right, bottom-left")
    return quad.astype(np.float32)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = polygon_array(points)
    offsets = points - points.mean(axis=0)
    cycle = points[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]
    candidates = [np.roll(cycle, -index, axis=0) for index in range(4)]

    def axis_alignment(quad: np.ndarray) -> float:
        rightward = (quad[1] - quad[0]) + (quad[2] - quad[3])
        downward = (quad[3] - quad[0]) + (quad[2] - quad[1])
        return float(rightward[0] / max(np.linalg.norm(rightward), 1) +
                     downward[1] / max(np.linalg.norm(downward), 1))

    return max(candidates, key=axis_alignment)


def load_annotation(path: Path, photo: Photo) -> Annotation:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise DocscanError("invalid_corners", "Expected a JSON object")
        space = data.get("coordinate_space")
        if space not in (NORMALIZED_SPACE, RAW_SPACE) or data.get("order") != CORNER_ORDER:
            raise DocscanError("invalid_corners", f"Specify coordinate_space and order={CORNER_ORDER}")
        size = photo.raw_size if space == RAW_SPACE else photo.size
        declared = data.get("image_size")
        if not isinstance(declared, list) or any(type(value) is not int for value in declared) or declared != list(size):
            raise DocscanError("invalid_corners", f"image_size {declared!r} does not match {list(size)} in {space}")
        points = validate_quad(data.get("points"), size)
        shapes = data.get("text_polygons", [])
        if not isinstance(shapes, list):
            raise DocscanError("invalid_corners", "text_polygons must be a list of quadrilaterals")
        polygons = tuple(polygon_array(shape) for shape in shapes)
        if any(np.any(shape < 0) or np.any(shape > np.asarray(size) - 1) for shape in polygons):
            raise DocscanError("invalid_corners", "A text polygon is outside the image")
        if space == RAW_SPACE:
            points = validate_quad(order_quad(raw_to_upright(points, photo)), photo.size)
            polygons = tuple(raw_to_upright(shape, photo) for shape in polygons)
        source = data.get("source")
        if source not in (None, "real", "synthetic"):
            raise DocscanError("invalid_corners", "source must be real or synthetic when supplied")
        return Annotation(points, polygons, source, space)
    except DocscanError as error:
        raise DocscanError(error.code, f"{path}: {error}") from error
    except (OSError, UnicodeError, ValueError) as error:
        raise DocscanError("invalid_corners", f"Cannot read {path}: {error}") from error


@dataclass(frozen=True)
class Detection:
    corners: np.ndarray | None
    diagnostics: dict
    partial_sides: tuple[dict, ...] = ()


def _edge_bands(gray: np.ndarray, chroma: np.ndarray, points: np.ndarray,
                inward: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape
    inner = np.rint(points + inward[:, None, :] * offset).astype(int)
    outer = np.rint(points - inward[:, None, :] * offset).astype(int)
    limits = np.array([width - 1, height - 1])
    valid = np.all((inner >= 0) & (inner <= limits) & (outer >= 0) & (outer <= limits), axis=-1)
    ix, iy = np.moveaxis(np.clip(inner, 0, limits), -1, 0)
    ox, oy = np.moveaxis(np.clip(outer, 0, limits), -1, 0)
    light = gray[iy, ix].astype(np.int16) - gray[oy, ox]
    neutral = chroma[oy, ox].astype(np.int16) - chroma[iy, ix]
    paper = (light >= 8) | (neutral >= 6)
    contrast = (np.abs(light) >= 6) | (np.abs(neutral) >= 5)
    return (valid & paper).mean(axis=1), (valid & contrast).mean(axis=1), valid.mean(axis=1)


def _side_lines(edges: np.ndarray, gray: np.ndarray, chroma: np.ndarray, offset: float) -> list[list[np.ndarray]]:
    height, width = gray.shape
    short = min(height, width)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(16, round(0.11 * short)),
                               minLineLength=max(18, round(0.17 * short)), maxLineGap=round(0.045 * short))
    if segments is None:
        return [[], [], [], []]
    groups = [[], [], [], []]  # top, right, bottom, left
    for segment in segments[:200, 0]:
        start, end = segment.astype(float).reshape(2, 2)
        delta = end - start
        horizontal = abs(delta[0]) >= abs(delta[1])
        axis = 0 if horizontal else 1
        if delta[axis] < 0:
            start, end, delta = end, start, -delta
        if delta[axis] == 0:
            continue
        slope = delta[1 - axis] / delta[axis]
        center = width / 2 if horizontal else height / 2
        position = start[1 - axis] + (center - start[axis]) * slope
        limit = height if horizontal else width
        if not 0 <= position < limit:
            continue
        near_side = position < limit / 2
        side = (0 if near_side else 2) if horizontal else (3 if near_side else 1)
        length = float(np.linalg.norm(delta))
        inward = np.array([-delta[1], delta[0]]) / length
        if side in (2, 3):
            inward = -inward
        samples = start + np.linspace(0.08, 0.92, 17)[:, None] * delta
        votes, contrast, context = _edge_bands(gray, chroma, samples[None], inward[None], offset)
        if context[0] < 0.60 or contrast[0] < 0.25:
            continue
        rank = 0.60 * votes[0] + 0.25 * contrast[0] + 0.15 * min(length / short, 1)
        line = np.cross(np.r_[start, 1], np.r_[end, 1]) / length
        groups[side].append((rank, position, slope, line))

    sides = []
    for candidates in groups:
        unique = []
        for _, position, slope, line in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(abs(position - other_pos) > 0.012 * short or abs(slope - other_slope) > 0.04
                   for other_pos, other_slope, _ in unique):
                unique.append((position, slope, line))
            if len(unique) == 4:
                break
        sides.append([line for _, _, line in unique])
    return sides


def _line_quads(sides: list[list[np.ndarray]]):
    for top, right, bottom, left in product(*sides):
        intersections = np.array([np.cross(top, left), np.cross(top, right),
                                  np.cross(bottom, right), np.cross(bottom, left)])
        if np.all(np.abs(intersections[:, 2]) > 0.1):
            yield intersections[:, :2] / intersections[:, 2:]


def _partial_sides(sides: list[list[np.ndarray]], gray: np.ndarray, chroma: np.ndarray,
                   distance: np.ndarray, offset: float, original: tuple[int, int]) -> tuple[dict, ...]:
    height, width = gray.shape
    center = np.array([width / 2, height / 2, 1])
    scale = np.array([width / original[0], height / original[1]])
    fractions = np.linspace(0.10, 0.87, 39)
    output = []
    for side, (name, lines) in enumerate(zip(("top", "right", "bottom", "left"), sides, strict=True)):
        horizontal = side % 2 == 0
        axis = 0 if horizontal else 1
        extent = width if horizontal else height
        other_extent = height if horizontal else width
        abscissa = fractions * (extent - 1)
        best = None
        for raw_line in lines:
            line = raw_line if raw_line @ center > 0 else -raw_line
            if abs(line[1 - axis]) < 0.88:
                continue
            across = -(line[axis] * abscissa + line[2]) / line[1 - axis]
            position = float(across.mean()) / other_extent
            if (side in (0, 3) and position > 0.34) or (side in (1, 2) and position < 0.66):
                continue
            samples = np.column_stack((abscissa, across) if horizontal else (across, abscissa))
            paper, contrast, context = _edge_bands(gray, chroma, samples[None], line[None, :2], offset)
            wider, _, wide_context = _edge_bands(gray, chroma, samples[None], line[None, :2], 2.5 * offset)
            x, y = np.moveaxis(np.rint(np.clip(samples, 0, [width - 1, height - 1])).astype(int), -1, 0)
            support = float((distance[y, x] <= max(3.5, 0.008 * min(height, width))).mean())
            if (paper[0] < 0.52 or contrast[0] < 0.56 or context[0] < 0.65 or support < 0.57 or
                    wider[0] < 0.58 or wide_context[0] < 0.65):
                continue
            inside = samples + line[:2] * offset
            ix, iy = np.moveaxis(np.rint(np.clip(inside, 0, [width - 1, height - 1])).astype(int), -1, 0)
            neutral = 1 - min(float(np.median(chroma[iy, ix])) / 55, 1)
            score = float(0.42 * paper[0] + 0.37 * support + 0.21 * neutral)
            if best is None or score > best[0]:
                mapped = np.r_[line[:2] * scale, line[2] + line[:2] @ (0.5 * scale - 0.5)]
                best = (score, {"side": name, "line": mapped.tolist(), "support": support,
                                "paper_transition": float(paper[0])})
        if best is not None:
            output.append(best[1])
    return tuple(output)


def _contour_quads(masks: tuple, min_area: float):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for kind, mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            if cv2.contourArea(contour) < min_area:
                break
            perimeter = cv2.arcLength(contour, True)
            for fraction in (0.012, 0.02, 0.03, 0.045):
                vertices = cv2.approxPolyDP(contour, fraction * perimeter, True)
                if len(vertices) == 4 and cv2.isContourConvex(vertices):
                    yield kind, order_quad(vertices[:, 0, :])
                    break


def detect_page(bgr: np.ndarray, config: dict, orientation: str, *, partial: bool = False) -> Detection:
    height, width = bgr.shape[:2]
    shrink = min(1, config["detect_max_side"] / max(width, height))
    dw, dh = max(1, round(width * shrink)), max(1, round(height * shrink))
    if min(dw, dh) < 8:
        return Detection(None, {"reason": "Image too small for contour detection", "detection_size": [dw, dh]})
    small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA) if shrink < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    local = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(local, 32, 92)
    chroma = cv2.GaussianBlur(np.ptp(small, axis=2), (5, 5), 0)
    chroma_edges = cv2.Canny(chroma, 8, 20)
    combined = cv2.bitwise_or(edges, chroma_edges)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, neutral = cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    distance = cv2.distanceTransform(cv2.bitwise_not(combined), cv2.DIST_L2, 3)
    short = min(dw, dh)
    offset = max(3.0, 0.015 * short)
    expected_aspect = 210 / 297 if orientation == "portrait" else 297 / 210
    masks = (("edges", combined), ("bright", bright), ("low_chroma", neutral))
    candidates = list(_contour_quads(masks, 0.55 * config["min_area_ratio"] * dw * dh))
    sides = _side_lines(combined, gray, chroma, offset)
    candidates.extend(("fitted_lines", quad) for quad in _line_quads(sides))
    partial_sides = _partial_sides(sides, gray, chroma, distance, offset, (width, height)) if partial else ()
    best = None
    accepted = 0
    fractions = np.linspace(0.11, 0.89, 29)[None, :, None]
    for kind, points in candidates:
        try:
            quad = validate_quad(points, (dw, dh))
        except DocscanError:
            continue
        area = signed_area(quad) / (dw * dh)
        if not config["min_area_ratio"] <= area <= config["max_area_ratio"]:
            continue
        vectors = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(vectors, axis=1)
        if min(lengths) < 0.10 * short:
            continue
        opposing = lengths.reshape(2, 2)
        if np.any(opposing.min(axis=0) < 0.42 * opposing.max(axis=0)):
            continue
        unit = vectors / lengths[:, None]
        inward = np.column_stack((-unit[:, 1], unit[:, 0]))
        samples = quad[:, None, :] + vectors[:, None, :] * fractions
        sx, sy = np.moveaxis(np.rint(samples).astype(int), -1, 0)
        support = (distance[sy, sx] <= max(3.5, 0.007 * short)).mean(axis=1)
        paper, contrast, context = _edge_bands(gray, chroma, samples, inward, offset)
        if (context.min() < 0.65 or support.min() < 0.34 or contrast.min() < 0.32 or
                paper.mean() < 0.46 or np.count_nonzero(paper >= 0.38) < 3):
            continue
        aspect = (lengths[0] + lengths[2]) / (lengths[1] + lengths[3])
        aspect_score = min(aspect / expected_aspect, expected_aspect / aspect)
        angle_score = 1 - np.abs(np.sum(unit * np.roll(unit, -1, axis=0), axis=1)).mean()
        score = float(0.37 * paper.mean() + 0.15 * paper.min() + 0.24 * support.mean() +
                      0.12 * support.min() + 0.06 * angle_score + 0.04 * aspect_score + 0.02 * area)
        accepted += 1
        if best is None or score > best[0]:
            best = (score, quad, kind, area, support, paper)
    diagnostics = {"detection_size": [dw, dh], "candidate_count": accepted, "proposed_count": len(candidates)}
    if best is None:
        return Detection(None, {**diagnostics, "reason": "No page candidate had supported edges and visible surrounding context"}, partial_sides)
    score, points, kind, area, support, paper = best
    mapped = (points + 0.5) * [width / dw, height / dh] - 0.5
    return Detection(validate_quad(mapped, (width, height)), {**diagnostics, "candidate_source": kind,
                     "score": score, "area_ratio": area, "boundary_support": float(support.mean()),
                     "edge_support_by_side": support.tolist(), "paper_transition_by_side": paper.tolist()}, partial_sides)


def projected_page_aspect(quad: np.ndarray, image_size: tuple[int, int]) -> float | None:
    """Recover a rectangular page's aspect under a centered, square-pixel camera."""
    unit_page = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    projection = cv2.getPerspectiveTransform(unit_page, np.float32(quad))
    center = (np.asarray(image_size) - 1) / 2
    projection[:2] -= center[:, None] * projection[2]
    horizontal, vertical = projection[:, 0], projection[:, 1]
    if min(abs(horizontal[2]), abs(vertical[2])) < 0.012:
        return None

    # In camera coordinates the two printed page axes are orthogonal.
    focal_squared = -np.dot(horizontal[:2], vertical[:2]) / (horizontal[2] * vertical[2])
    diagonal_squared = float(np.dot(image_size, image_size))
    if not 0.28**2 * diagonal_squared <= focal_squared <= 2.5**2 * diagonal_squared:
        return None
    inverse_camera = np.array([1, 1, np.sqrt(focal_squared)])
    aspect = float(np.linalg.norm(horizontal * inverse_camera) / np.linalg.norm(vertical * inverse_camera))
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    apparent = max(sides[0], sides[2]) / max(sides[1], sides[3])
    return aspect if 0.68 <= aspect / apparent <= 1.45 else None


def rectified_size(quad: np.ndarray, config: dict, orientation: str, *,
                   aspect: float | None = None) -> tuple[int, int]:
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    observed_width, observed_height = max(lengths[0], lengths[2]), max(lengths[1], lengths[3])
    if config["page_format"] == "a4":
        reference_width, reference_height = (210, 297) if orientation == "portrait" else (297, 210)
        scale = min(observed_width / reference_width, observed_height / reference_height,
                    config["max_long_edge"] / max(reference_width, reference_height))
        return round(scale * reference_width), round(scale * reference_height)
    if aspect is not None:
        observed_height = max(observed_height, observed_width / aspect)
        observed_width = observed_height * aspect
    scale = min(1, config["max_long_edge"] / max(observed_width, observed_height))
    return round(scale * observed_width), round(scale * observed_height)


def warp_document(bgr: np.ndarray, corners: np.ndarray, config: dict, orientation: str, *,
                  interpolation: Literal["linear", "cubic"] = "linear",
                  output_size: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
    h, w = bgr.shape[:2]
    quad = validate_quad(corners, (w, h))
    ww, hh = output_size if output_size is not None else rectified_size(quad, config, orientation)
    if min(ww, hh) < 3:
        raise DocscanError("warp_too_small", f"Estimated page is too small: {[ww, hh]}")
    target = np.float32([[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]])
    matrix = cv2.getPerspectiveTransform(quad, target)
    kernel = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[interpolation]
    warped = cv2.warpPerspective(bgr, matrix, (ww, hh), flags=kernel, borderMode=cv2.BORDER_REPLICATE)
    return warped, {
        "input_coordinate_space": NORMALIZED_SPACE, "order": CORNER_ORDER, "points": quad.tolist(),
        "input_size": [w, h], "output_size": [ww, hh], "page_format": config["page_format"],
        "page_orientation": orientation, "input_to_rectified": matrix.tolist(),
        "rectified_to_input": np.linalg.inv(matrix).tolist(), "interpolation": interpolation,
    }


def corner_overlay(bgr: np.ndarray, corners: np.ndarray | None, label: str) -> np.ndarray:
    view = bgr.copy()
    thickness = max(1, round(max(bgr.shape[:2]) / 450))
    color = (10, 190, 30) if label == "auto" else (30, 136, 255)
    if corners is not None:
        quad = np.rint(corners).astype(np.int32)
        cv2.polylines(view, [quad], True, color, thickness, cv2.LINE_AA)
        for index, (x, y) in enumerate(quad, start=1):
            cv2.circle(view, (x, y), 3 * thickness, color, -1, cv2.LINE_AA)
            cv2.putText(view, str(index), (x + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)
    cv2.putText(view, label, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, max(thickness, 2), cv2.LINE_AA)
    return view
