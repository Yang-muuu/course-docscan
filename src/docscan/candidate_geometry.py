"""Find a whole sheet, or the dominant leaf next to a visible binding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from .enhance import gray_image
from .geometry import detect_page


@dataclass(frozen=True)
class PageGeometry:
    corners: np.ndarray
    text_angle: float
    spine_side: int | None
    mode: str


def _text_angle(polygons: np.ndarray) -> float:
    if not len(polygons):
        return 0.0
    edges = np.roll(polygons, -1, axis=1) - polygons
    lengths = np.linalg.norm(edges, axis=-1)
    first, second = lengths[:, ::2].mean(axis=1), lengths[:, 1::2].mean(axis=1)
    along_first = first >= second
    direction = np.where(along_first[:, None], edges[:, 0] - edges[:, 2], edges[:, 1] - edges[:, 3])
    longer, shorter = np.maximum(first, second), np.minimum(first, second)
    lines = (longer >= 35) & (longer > 2.5 * np.maximum(shorter, 1))
    angles = np.arctan2(direction[lines, 1], direction[lines, 0])
    weights = longer[lines] ** 0.75
    phase = np.sum(weights * np.exp(2j * angles))
    return float(np.angle(phase) * 90 / np.pi) if abs(phase) > 0.55 * weights.sum() else 0.0


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    coordinates = np.rint(points).astype(np.int32)
    x, y = np.moveaxis(np.clip(coordinates, 0, [image.shape[1] - 1, image.shape[0] - 1]), -1, 0)
    return image[y, x]


def _paper_transition(gray, saturation, valid, inside, outside, *, light_floor, light_range, tint_floor, tint_range):
    light = _sample(gray, inside).astype(np.float32) - _sample(gray, outside)
    tint = _sample(saturation, outside) - _sample(saturation, inside)
    active = (_sample(valid, inside) > 0) & (_sample(valid, outside) > 0)
    return np.maximum(np.clip((light - light_floor) / light_range, 0, 1),
                      np.clip((tint - tint_floor) / tint_range, 0, 1)) * active


def _lines(gray: np.ndarray, saturation: np.ndarray, valid: np.ndarray):
    height, width = gray.shape
    short = min(height, width)
    center = np.array([width, height]) / 2
    local = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.bitwise_or(cv2.Canny(local, 22, 65), cv2.Canny(np.uint8(255 * saturation), 10, 31))
    vertical = cv2.Canny(cv2.GaussianBlur(gray, (3, 25), 0), 8, 22)
    horizontal = cv2.Canny(cv2.GaussianBlur(gray, (25, 3), 0), 8, 22)
    safe = cv2.erode(valid, np.ones((9, 9), np.uint8)) != 0
    proposals = [[], [], [], []]
    for layer, preferred_axis in ((edges, None), (vertical, 1), (horizontal, 0)):
        layer[~safe] = 0
        segments = cv2.HoughLinesP(layer, 1, np.pi / 360, threshold=round(0.075 * short),
                                   minLineLength=round(0.17 * short), maxLineGap=round(0.05 * short))
        if segments is None:
            continue
        for segment in segments[:, 0]:
            start, end = segment.reshape(2, 2).astype(float)
            vector = end - start
            axis = int(abs(vector[1]) > abs(vector[0]))
            if preferred_axis is not None and axis != preferred_axis:
                continue
            slope = vector[1 - axis] / vector[axis]
            if abs(slope) > 0.33:
                continue
            length = float(np.linalg.norm(vector))
            line = np.cross(np.r_[start, 1], np.r_[end, 1]) / length
            position = start[1 - axis] + slope * (center[axis] - start[axis])
            points = start + np.linspace(0.07, 0.93, 23)[:, None] * vector
            for side in ((0, 2) if axis == 0 else (1, 3)):
                sign = 1 if side in (0, 3) else -1
                if sign * (position - center[1 - axis]) > 0.14 * short:
                    continue
                inward = line[:2] * sign * np.sign(line[1 - axis])
                transition = 0.0
                for offset in (0.012, 0.036):
                    delta = inward * (offset * short)
                    transition += _paper_transition(gray, saturation, valid, points + delta, points - delta,
                                                    light_floor=6, light_range=65, tint_floor=0.025, tint_range=0.20).mean()
                score = 0.46 * transition + 0.18 * min(1, length / short)
                proposals[side].append((score, length, position, slope, line, preferred_axis is not None))

    groups = []
    for group in proposals:
        kept = []
        structured = [item for item in group if item[5]]
        for limit, options, rank in ((6, group, 0), (10, structured, 1), (12, group, 1)):
            for _, _, position, slope, line, _ in sorted(options, key=lambda item: item[rank], reverse=True)[:45]:
                if len(kept) >= limit:
                    break
                if all(abs(position - p) >= 0.009 * short or abs(slope - s) >= 0.033 for p, s, _ in kept):
                    kept.append((position, slope, line))
        groups.append([line for _, _, line in kept])
    return cv2.bitwise_or(edges, cv2.bitwise_or(vertical, horizontal)), vertical, groups


def _binding(gray: np.ndarray, saturation: np.ndarray, valid: np.ndarray, text: np.ndarray) -> np.ndarray | None:
    height, width = gray.shape
    short = min(height, width)
    if len(text) < 10:
        return None
    centers, half = text.mean(axis=1), np.ptp(text, axis=1) / 2
    low, high = np.quantile(centers[:, 1], [0.035, 0.965])
    span, mid_y = high - low, (high + low) / 2
    if span < 0.22 * short:
        return None
    rows = np.rint(np.linspace(low + 0.06 * span, high - 0.06 * span, 84)).astype(int)
    positions, slopes = np.meshgrid(np.arange(0.16 * width, 0.84 * width, max(2, round(short * 0.004))),
                                    np.linspace(-0.32, 0.32, 65))
    positions, slopes = positions.ravel(), slopes.ravel()
    cols = np.rint(positions[:, None] + slopes[:, None] * (rows - mid_y)).astype(int)
    radius = max(3, round(short * 0.007))
    cols = np.clip(cols, 2 * radius, width - 1 - 2 * radius)
    # Smooth along the binding so character strokes do not vote for a page split.
    smooth = cv2.GaussianBlur(gray, (3, 19), 0)
    gradient = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)) / 8
    peak = cv2.dilate(gradient, np.ones((1, max(5, 2 * round(short * 0.007) + 1)), np.uint8))[rows, cols]
    offset = max(10, round(0.024 * short))
    left, right = np.clip(cols - offset, 0, width - 1), np.clip(cols + offset, 0, width - 1)
    luma_left, luma_right = smooth[rows, left], smooth[rows, right]
    paper = ((luma_left > 45) & (luma_right > 45) & (saturation[rows, left] < 0.29)
             & (saturation[rows, right] < 0.29) & (valid[rows, left] > 0) & (valid[rows, right] > 0))
    relative = peak / np.maximum(60, np.maximum(luma_left, luma_right))
    continuity = ((relative > 0.011) & paper).mean(axis=1)
    strength = (np.clip((relative - 0.004) / 0.046, 0, 1) * paper).mean(axis=1)
    depth = centers[:, 0] - (positions[:, None] + slopes[:, None] * (centers[:, 1] - mid_y))
    inside = (centers[:, 1] >= low - 0.05 * short) & (centers[:, 1] <= high + 0.05 * short)
    neighbors = (np.abs(depth) < 0.27 * short) & inside
    before = (neighbors & (depth < -0.02 * short)).sum(axis=1)
    after = (neighbors & (depth > 0.02 * short)).sum(axis=1)
    facing_print = np.minimum(np.minimum(before, after) / 10, 1)
    crossing = (np.abs(depth) < half[:, 0] + np.abs(slopes[:, None]) * half[:, 1] - 0.004 * short) & inside
    cuts = (crossing * np.clip(half[:, 0] / (0.05 * short), 0.10, 1)).sum(axis=1)
    valid_spines = (continuity > 0.54) & (paper.mean(axis=1) > 0.68) & (facing_print > 0.40) & (cuts < 8)
    score = 0.46 * strength + 0.33 * continuity + 0.17 * facing_print - 0.20 * np.minimum(cuts / 6, 1)
    best = int(np.argmax(np.where(valid_spines, score, -np.inf)))
    if not valid_spines[best] or score[best] < 0.38:
        return None
    return np.array([1, -slopes[best], -positions[best] + slopes[best] * mid_y])


def _book_margins(gray, saturation, valid, text, spine):
    """Search for the faint top and bottom edges beyond the wider printed field."""
    short = min(gray.shape)
    depth = text.mean(axis=1) @ spine[:2] + spine[2]
    sides = text[depth < 0], text[depth >= 0]
    if min(map(len, sides)) < 5:
        return [], []
    main = max(sides, key=lambda boxes: np.ptp(np.quantile(boxes[:, :, 0], [0.04, 0.96])))
    left, right = np.quantile(main[:, :, 0], [0.045, 0.955])
    width = right - left
    if width < 0.19 * short:
        return [], []
    columns = np.linspace(left + 0.06 * width, right - 0.06 * width, 45)
    mid_x = (left + right) / 2
    groups = []
    for sign, quantile in ((1, 0.012), (-1, 0.988)):
        edge = np.quantile(main[:, :, 1], quantile)
        levels, slopes = np.meshgrid(edge - sign * np.arange(0.007 * short, 0.14 * short, max(1, round(short * 0.003))),
                                     np.linspace(-0.30, 0.30, 41))
        levels, slopes = levels.ravel(), slopes.ravel()
        ys = levels[:, None] + slopes[:, None] * (columns - mid_x)
        points = np.stack((np.broadcast_to(columns, ys.shape), ys), axis=-1)
        transition = np.zeros(len(points))
        for offset in (0.008, 0.023):
            delta = [0, sign * offset * short]
            transition += _paper_transition(gray, saturation, valid, points + delta, points - delta,
                                            light_floor=2, light_range=43, tint_floor=0.013, tint_range=0.15).mean(axis=1)
        across = levels[:, None, None] + slopes[:, None, None] * (main[None, :, :, 0] - mid_x)
        clear = (np.min(sign * (main[None, :, :, 1] - across), axis=2) > -0.002 * short).mean(axis=1)
        score = 0.50 * transition + 0.55 * clear - 0.10 * np.abs(levels - edge) / short
        selected = []
        for index in np.argsort(score)[::-1]:
            level, slope = levels[index], slopes[index]
            if all(abs(level - y) > 0.008 * short or abs(slope - m) > 0.045 for y, m in selected):
                selected.append((level, slope))
            if len(selected) == 4:
                break
        groups.append([np.array([-slope, 1, -level + slope * mid_x]) for level, slope in selected])
    return groups


def _text_fit(text, corners, inward, short):
    centers, half = text.mean(axis=1), np.ptp(text, axis=1) / 2
    depth = np.einsum("nsk,pk->nsp", inward, centers) - np.sum(corners * inward, axis=-1)[:, :, None]
    radius = np.einsum("nsk,pk->nsp", np.abs(inward), half)
    contained = (depth >= -0.007 * short).all(axis=1)
    weights = np.clip(half[:, 0] / (0.035 * short), 0.18, 1)
    cuts, facing = [], []
    for side in range(4):
        rest = np.delete(depth, side, axis=1).min(axis=1)
        crossing = (rest > -0.025 * short) & (np.abs(depth[:, side]) < radius[:, side] - 0.007 * short)
        cuts.append((crossing * weights).sum(axis=1))
        beyond = (rest > -0.06 * short) & (depth[:, side] < -0.012 * short) & (depth[:, side] > -0.28 * short)
        facing.append(np.minimum(beyond.sum(axis=1) / 12, 1))
    return depth, contained, np.stack(cuts, axis=1), np.stack(facing, axis=1)


def _choose_quad(gray, saturation, valid, text, edges, vertical, groups, spine):
    height, width = gray.shape
    short = min(height, width)
    valid_area = np.count_nonzero(valid)
    lines = np.array(list(product(*groups)))
    joints = np.cross(lines, np.roll(lines, 1, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        quads = joints[:, :, :2] / joints[:, :, 2, None]
    usable = np.isfinite(quads).all(axis=(1, 2))
    quads, lines = quads[usable], lines[usable]
    vectors = np.roll(quads, -1, axis=1) - quads
    lengths = np.linalg.norm(vectors, axis=-1)
    aspect = lengths[:, ::2].mean(axis=1) / np.maximum(lengths[:, 1::2].mean(axis=1), 1e-5)
    x, y = np.moveaxis(quads, -1, 0)
    area = (x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1) / (2 * valid_area)
    usable = ((lengths > 0.23 * short).all(axis=1) & (aspect > 0.45) & (aspect < 1.10) & (area > 0.20)
              & (x.min(axis=1) > -0.08 * short) & (x.max(axis=1) < width + 0.08 * short)
              & (y.min(axis=1) > -0.08 * short) & (y.max(axis=1) < height + 0.08 * short)
              & (vectors[:, 0, 0] > 0) & (vectors[:, 1, 1] > 0) & (vectors[:, 2, 0] < 0) & (vectors[:, 3, 1] < 0))
    quads, lines, vectors, lengths, aspect, area = (values[usable] for values in (quads, lines, vectors, lengths, aspect, area))
    if not len(quads):
        return None

    points = quads[:, :, None] + vectors[:, :, None] * np.linspace(0.10, 0.90, 41)[None, None, :, None]
    inward = np.stack((-vectors[:, :, 1], vectors[:, :, 0]), axis=-1) / lengths[:, :, None]
    distance = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    support = (_sample(distance, points) < max(2.5, 0.006 * short)).mean(axis=2)
    crease = cv2.distanceTransform(cv2.bitwise_not(vertical), cv2.DIST_L2, 3)
    binding_support = (_sample(crease, points) < max(3, 0.008 * short)).mean(axis=2)
    delta = inward[:, :, None] * (0.017 * short)
    inside, outside = points + delta, points - delta
    exterior = _paper_transition(gray, saturation, valid, inside, outside, light_floor=3, light_range=54,
                                 tint_floor=0.022, tint_range=0.16).mean(axis=2)
    two_sheets = ((_sample(gray, inside) > 43) & (_sample(gray, outside) > 43)
                  & (_sample(saturation, inside) < 0.25) & (_sample(saturation, outside) < 0.25)
                  & (_sample(valid, inside) > 0) & (_sample(valid, outside) > 0)).mean(axis=2)
    depth, contained, cuts, facing = _text_fit(text, quads, inward, short)
    text_fraction = contained.sum(axis=1) / max(1, len(text))
    common = (0.25 * support.mean(axis=1) + 0.13 * support.min(axis=1) + 0.22 * area + 0.08 * text_fraction
              - 0.20 * np.minimum(cuts.sum(axis=1) / 10, 1))
    paper_shape = np.exp(-(np.log(aspect / 0.715) / 0.28) ** 2)
    sheet = common + 0.37 * exterior.mean(axis=1) + 0.17 * exterior.min(axis=1) + 0.09 * paper_shape
    book_shape = np.exp(-(np.log(aspect / 0.67) / 0.22) ** 2)
    books = []
    for side in (1, 3):
        other_edges = np.delete(exterior, side, axis=1)
        book = (common + 0.33 * other_edges.mean(axis=1) + 0.13 * other_edges.min(axis=1) + 0.09 * book_shape
                + 0.18 * two_sheets[:, side] * binding_support[:, side] * facing[:, side] - 0.05 * exterior[:, side]
                - 0.17 * np.minimum(cuts[:, side] / 5, 1))
        possible = ((support[:, side] > 0.52) & (binding_support[:, side] > 0.45) & (two_sheets[:, side] > 0.47)
                    & (facing[:, side] > 0.23) & (other_edges.min(axis=1) > 0.08) & (book_shape > 0.21))
        if spine is not None:
            same_line = np.isclose(lines[:, side], spine, atol=1e-8).all(axis=1)
            main = depth[:, side] >= -0.004 * short
            coverage = (contained & main).sum(axis=1) / np.maximum(main.sum(axis=1), 1)
            book += 0.63 * coverage + 0.12 * area
            possible = same_line & (coverage > 0.79) & (book_shape > 0.17) & (other_edges.mean(axis=1) > 0.29)
        books.append(np.where(possible, book, -1))
    sheet_index = int(np.argmax(sheet))
    scores = np.stack(books, axis=1)
    book_index, binding_index = np.unravel_index(np.argmax(scores), scores.shape)
    book_score = scores[book_index, binding_index]
    if book_score > 0 and (spine is not None or book_score > sheet[sheet_index] + 0.026 or aspect[sheet_index] > 0.79):
        return quads[book_index], (1, 3)[binding_index]
    return (quads[sheet_index], None) if sheet[sheet_index] >= 0.63 else None


def locate_document(image: np.ndarray, text: np.ndarray, settings: dict) -> PageGeometry:
    height, width = image.shape[:2]
    angle = _text_angle(text)
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
    frame = np.ceil(np.abs(rotation[:, :2]) @ [width, height]).astype(int)
    rotation[:, 2] += (frame - [width, height]) / 2
    scale = min(1, 980 / max(frame))
    rotation *= scale
    size = tuple(np.rint(frame * scale).astype(int))
    small = cv2.warpAffine(image, rotation, size, borderValue=(1, 1, 1))
    valid = cv2.warpAffine(np.full((height, width), 255, np.uint8), rotation, size)
    text = text @ rotation[:, :2].T + rotation[:, 2]
    if min(size) >= 48:
        gray = cv2.GaussianBlur(gray_image(small), (5, 5), 0)
        saturation = np.ptp(small, axis=2) / (small.max(axis=2).astype(np.float32) + 24)
        saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
        edges, vertical, groups = _lines(gray, saturation, valid)
        spine = _binding(gray, saturation, valid, text)
        if spine is not None:
            groups[1].append(spine)
            groups[3].append(spine)
            top, bottom = _book_margins(gray, saturation, valid, text, spine)
            groups[0].extend(top)
            groups[2].extend(bottom)
        selected = _choose_quad(gray, saturation, valid, text, edges, vertical, groups, spine) if all(groups) else None
        if selected is not None:
            quad, side = selected
            corners = cv2.transform(np.float32(quad)[None], cv2.invertAffineTransform(rotation))[0]
            return PageGeometry(corners, angle, side, "book_leaf" if side is not None else "sheet")

    # Keep the project's existing contour detector for unusual page shapes and blank sheets.
    orientation = "portrait" if height >= width else "landscape"
    detection = detect_page(image, settings, orientation)
    if detection.corners is not None:
        return PageGeometry(detection.corners, angle, None, "outer_contour")
    full = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return PageGeometry(full, angle, None, "full_frame_review")
