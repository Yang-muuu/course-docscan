from __future__ import annotations

import cv2
import numpy as np

from .enhance import gray_image


def _printed_rules(ink: np.ndarray, distance: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                   depth: np.ndarray, short: int) -> np.ndarray:
    rules = np.zeros_like(ink)
    lines = cv2.HoughLinesP(ink * (depth <= 0.19 * short), 1, np.pi / 180,
                            threshold=max(18, round(0.053 * short)),
                            minLineLength=max(25, round(0.09 * short)), maxLineGap=max(3, round(0.007 * short)))
    if lines is None:
        return rules

    support = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    for ax, ay, bx, by in lines[:, 0]:
        sx, sy = bx - ax, by - ay
        if min(abs(sx), abs(sy)) > 0.057 * max(abs(sx), abs(sy)):
            continue
        xs = np.rint(np.linspace(ax, bx, 60)).astype(int)
        ys = np.rint(np.linspace(ay, by, 60)).astype(int)
        normal = dy[ys, 0] if abs(sx) > abs(sy) else dx[0, xs]
        if (not 0.011 * short < np.median(normal) < 0.18 * short or
                support[ys, xs].mean() < 0.94 or np.median(distance[ys, xs]) < 0.014 * short):
            continue
        # Extend the exclusion across a fold so the ruled line cannot connect it to the page interior.
        extension = 0.13 * short / np.hypot(sx, sy)
        offset_x, offset_y = round(sx * extension), round(sy * extension)
        cv2.line(rules, (ax - offset_x, ay - offset_y), (bx + offset_x, by + offset_y),
                 1, max(3, round(short * 0.004)) | 1)
    return rules


def clean_page_edges(image: np.ndarray, paper: np.ndarray, artwork: np.ndarray) -> np.ndarray:
    """White out stains connected to a complete page's photographed edge."""
    h, w = image.shape[:2]
    scale = min(1, 1650 / max(h, w))
    size = round(w * scale), round(h * scale)
    gray = cv2.resize(gray_image(image), size, interpolation=cv2.INTER_AREA)
    visible = cv2.resize(paper.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    protected_art = cv2.resize(artwork, size, interpolation=cv2.INTER_NEAREST) != 0
    sh, sw = gray.shape
    short = min(sh, sw)
    if short < 80:
        return image
    yy, xx = np.ogrid[:sh, :sw]
    dx = np.minimum(xx, sw - 1 - xx)
    dy = np.minimum(yy, sh - 1 - yy)
    depth = np.minimum(dx, dy)
    distance = cv2.distanceTransform(np.pad(visible, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    ink = np.uint8((gray < 229) & (visible != 0) & ~protected_art)
    rules = _printed_rules(ink, distance, dx, dy, depth, short)
    kernel = np.ones((3, 3), np.uint8)
    barrier = cv2.dilate(rules, kernel) != 0

    # Find the thin strip following the cut edge independently of any marks connected to it.
    band = np.uint8((depth <= 0.058 * short) & (distance <= 0.023 * short) & (ink != 0) & ~barrier)
    closed = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clear = np.zeros_like(ink)
    for k in np.unique(labels[(distance <= 0.007 * short) & (labels > 0)]):
        x, y, cw, ch, _ = stats[k]
        roi = np.s_[y:y + ch, x:x + cw]
        region = labels[roi] == k
        length, width = max(cw, ch), min(cw, ch)
        if (length > 0.055 * short and length > 2.5 * width or
                length >= 3 and distance[roi][region].max() < 0.010 * short):
            clear[roi][region] = 1

    # After separating the cut edge and printed rules, follow folds and fibers rooted at the margin.
    touches_cut = cv2.dilate(clear, np.ones((9, 9), np.uint8)) != 0
    closed = cv2.morphologyEx(ink * (depth <= 0.20 * short), cv2.MORPH_CLOSE, kernel)
    graph = np.uint8((closed != 0) & ~barrier & (clear == 0))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(graph, connectivity=8)
    near_edge = (distance <= 0.023 * short) & (depth <= 0.049 * short)
    roots = set(np.unique(labels[(graph > 0) & (near_edge | touches_cut)]))
    corner_parts = [[] for _ in range(4)]
    for k, (x, y, cw, ch, area) in enumerate(stats[1:], 1):
        length, width = max(cw, ch), min(cw, ch)
        if area < 6 or length < 0.015 * short or length > 0.21 * short:
            continue
        roi = np.s_[y:y + ch, x:x + cw]
        in_corner = min(x + cw, sw - x) < 0.145 * short and min(y + ch, sh - y) < 0.129 * short
        if in_corner and (length >= 0.024 * short or length >= 2.2 * width):
            corner_parts[int(x > sw / 2) + 2 * int(y > sh / 2)].append((k, roi))
        elif (k in roots and length >= 0.052 * short and width < 0.047 * short and
                area / (cw * ch) < 0.34 and length > 2.8 * width):
            clear[roi][labels[roi] == k] = 1

    neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(0.05 * short) | 1,) * 2)
    for parts in corner_parts:
        if not any(k in roots for k, _ in parts):
            continue
        seeds = np.zeros_like(ink)
        for k, roi in parts:
            if k in roots:
                seeds[roi][labels[roi] == k] = 1
        nearby = cv2.dilate(seeds, neighborhood)
        for k, roi in parts:
            region = labels[roi] == k
            if np.any(nearby[roi][region]):
                clear[roi][region] = 1

    clear = cv2.dilate(clear, kernel)
    clear[(rules != 0) | protected_art] = 0
    image[cv2.resize(clear, (w, h), interpolation=cv2.INTER_NEAREST) != 0] = 255
    return image
