from __future__ import annotations

import cv2
import numpy as np

from .common import DocscanError
from .geometry import warp_document


def _ink(gray: np.ndarray) -> np.ndarray:
    short = min(gray.shape)
    block = max(15, round(0.05 * short) | 1)
    dark = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    _, height, area = stats[:, 2:].T
    # A long thin fold can be dark even when it is free of printed marks.
    thickness = area / np.maximum(height, 1)
    fold = (height > 0.22 * gray.shape[0]) & (thickness < 0.035 * gray.shape[1]) & (height > 4 * thickness)
    keep = (area >= 3) & ~fold
    keep[0] = False
    return keep[labels]


def find_gutter(image: np.ndarray) -> tuple[tuple[float, float] | None, dict]:
    height, width = image.shape[:2]
    if min(width, height) < 48 or width < 0.95 * height:
        return None, {"reason": "Insufficient landscape area to infer a spread"}
    scale = min(1, 800 / max(width, height))
    small = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    ink = _ink(gray)
    light = cv2.GaussianBlur(gray, (5, 9), 0).astype(np.int16)
    # Search straight but slanted gutter lines and ignore the top and bottom margins.
    mid, drift = np.meshgrid(np.linspace(0.31, 0.69, 77), np.linspace(-0.20, 0.20, 33))
    mid, drift = mid.ravel(), drift.ravel()
    rows = np.rint(np.linspace(0.12 * (height - 1), 0.86 * (height - 1), 120)).astype(int)
    positions = mid[:, None] + drift[:, None] * (rows[None, :] / (height - 1) - 0.5)
    cols = np.rint(positions * (width - 1)).astype(int)
    sums = np.pad(np.cumsum(ink[rows], axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    row_indexes = np.arange(len(rows))

    def density(lo: float, hi: float) -> np.ndarray:
        start = np.clip(cols + round(lo * width), 0, width - 1)
        end = np.clip(cols + round(hi * width), 1, width)
        return ((sums[row_indexes, end] - sums[row_indexes, start]) / np.maximum(end - start, 1)).mean(axis=1)

    center = density(-0.012, 0.012)
    left = density(-0.21, -0.037)
    right = density(0.037, 0.21)
    flank = np.minimum(left, right)
    radius = max(4, round(width * 0.027))
    surrounding = np.minimum(light[rows, cols - radius], light[rows, cols + radius])
    spine = ((surrounding - light[rows, cols] >= 8) & (surrounding >= 72)).mean(axis=1)
    safe = center <= 0.018
    dark = safe & (spine >= 0.55)
    gap = safe & (flank >= 0.025) & (center <= 0.24 * flank)
    if not np.any(dark | gap):
        return None, {"reason": "No continuous binding or clear gap between printed pages"}
    if np.any(dark):
        eligible = dark
        score = spine - 2 * center - 0.22 * np.abs(mid - 0.5) - 0.06 * np.abs(drift)
        method = "binding_shadow"
    else:
        eligible = gap
        score = flank - 2.4 * center + 0.08 * spine - 0.18 * np.abs(mid - 0.5) - 0.07 * np.abs(drift)
        method = "printed_page_gap"
    best = int(np.argmax(np.where(eligible, score, -np.inf)))
    gutter = (float(mid[best] - drift[best] / 2), float(mid[best] + drift[best] / 2))
    return gutter, {"method": method, "center_ink": float(center[best]), "flank_ink": float(flank[best]),
                    "shadow_support": float(spine[best])}


def split_spread(image: np.ndarray, gutter: tuple[float, float], geometry: dict):
    height, width = image.shape[:2]
    top, bottom = np.rint(np.asarray(gutter) * (width - 1)).astype(int)
    if height < 3 or min(top, bottom, width - 1 - top, width - 1 - bottom) < 3:
        raise DocscanError("invalid_gutter", "Each side of the binding needs at least three pixels")
    if top == bottom:
        return [("left", image[:, :top], {"crop": [0, 0, int(top), height]}),
                ("right", image[:, top:], {"crop": [int(top), 0, width - int(top), height]})]
    settings = {**geometry, "page_format": "native", "max_long_edge": max(height, width)}
    halves = (
        ("left", [[0, 0], [top, 0], [bottom, height - 1], [0, height - 1]]),
        ("right", [[top, 0], [width - 1, 0], [width - 1, height - 1], [bottom, height - 1]]),
    )
    return [(side, *warp_document(image, np.float32(points), settings, "portrait", interpolation="cubic"))
            for side, points in halves]
