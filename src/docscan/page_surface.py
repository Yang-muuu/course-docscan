from __future__ import annotations

import cv2
import numpy as np

from .common import DocscanError
from .enhance import gray_image
from .geometry import detect_page, projected_page_aspect, rectified_size, warp_document
from .page_margin import clean_page_edges


def _size(image: np.ndarray) -> tuple[int, int]:
    return image.shape[1], image.shape[0]


def _limited_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    ratio = min(1, max_side / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def resize_page(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = _size(image)
    if size == (width, height):
        return image
    interpolation = cv2.INTER_AREA if size[0] <= width and size[1] <= height else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    return resize_page(image, _limited_size(_size(image), max_side))


def pdf_raster_layout(nominal_size: tuple[int, int], page_format: str, orientation: str,
                      reference_dpi: float, dpi: int | None) -> tuple[tuple[float, float] | None, tuple[int, int]]:
    if dpi is None:
        return None, nominal_size
    if page_format == "a4":
        short, long = 210 * 72 / 25.4, 297 * 72 / 25.4
        width, height = (short, long) if orientation == "portrait" else (long, short)
    else:
        width, height = nominal_size[0] * 72 / reference_dpi, nominal_size[1] * 72 / reference_dpi
    pixels = max(3, round(width * dpi / 72)), max(3, round(height * dpi / 72))
    if max(pixels) > 12000 or pixels[0] * pixels[1] > 16_000_000:
        raise DocscanError("pdf_raster_too_large", f"Requested PDF raster is {pixels[0]} × {pixels[1]} pixels. Use a lower --dpi.")
    return (width, height), pixels


def _component(mask: np.ndarray, center_weight: bool = False) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        width, height = _size(mask)

        def score(contour):
            area = cv2.contourArea(contour)
            central = cv2.pointPolygonTest(contour, (width / 2, height / 2), False) >= 0
            return area * (1.8 if center_weight and central else 1)

        cv2.drawContours(filled, [max(contours, key=score)], -1, 255, cv2.FILLED)
    return filled


def _paper_seed(small: np.ndarray, close_size: int) -> np.ndarray:
    gray = gray_image(small).astype(np.float32)
    chroma = np.ptp(small, axis=2)
    # A white or off-white sheet is usually both lighter and less saturated than its surroundings.
    tone = np.clip(gray - 1.45 * chroma, 0, 255).astype(np.uint8)
    tone = cv2.medianBlur(tone, 5)
    _, mask = cv2.threshold(tone, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    return _component(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), center_weight=True)


def visible_paper(image: np.ndarray, corners: np.ndarray | None, sides: tuple[dict, ...] = (),
                  *, confirm_quad: bool = False) -> tuple[np.ndarray, dict]:
    width, height = _size(image)
    small = _resize(image, 720)
    sw, sh = _size(small)
    short = min(sw, sh)
    if short < 20:
        return np.full((height, width), 255, np.uint8), {"method": "unresolved", "reason": "Input too small for paper segmentation"}
    close_size = max(5, round(short * 0.025) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    tone_seed = _paper_seed(small, close_size)
    slices = {"top": np.s_[:3, :], "right": np.s_[:, -3:],
              "bottom": np.s_[-3:, :], "left": np.s_[:, :3]}
    edge_seed_coverage = {side: np.mean(tone_seed[region] != 0) for side, region in slices.items()}
    details = {}
    prior = np.zeros((sh, sw), np.uint8)
    if corners is not None:
        points = np.rint((corners + 0.5) * [sw / width, sh / height] - 0.5).astype(np.int32)
        cv2.fillConvexPoly(prior, points, 255)
        paper_pixels = cv2.countNonZero(tone_seed)
        outside = cv2.countNonZero(cv2.bitwise_and(tone_seed, cv2.bitwise_not(cv2.dilate(prior, kernel))))
        cuts_frame = any(edge_seed_coverage[side] >= 0.60 and np.mean(prior[region] != 0) < 0.25
                         for side, region in slices.items())
        cuts_region = paper_pixels > 0.25 * prior.size and outside > 0.22 * paper_pixels
        if confirm_quad and (cuts_frame or cuts_region):
            corners = None
            details["quad_rejected"] = "The visible paper region extends outside the proposed quadrilateral"
    bounds = np.full((sh, sw), 255, np.uint8)
    core_size = max(9, round(short * 0.16) | 1)
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size))
    if corners is not None:
        core = cv2.erode(prior, core_kernel)
        core[tone_seed == 0] = 0
        method = "edge_guided"
    else:
        sides = tuple(side for side in sides if edge_seed_coverage[side["side"]] < 0.68)
        if sides:
            xs = (np.arange(sw) + 0.5) * width / sw - 0.5
            ys = (np.arange(sh) + 0.5) * height / sh - 0.5
            for side in sides:
                a, b, c = side["line"]
                bounds[a * xs[None, :] + b * ys[:, None] + c < -3 * np.hypot(a, b)] = 0
            details["partial_sides"] = sides
        prior = _component(cv2.bitwise_and(tone_seed, bounds), center_weight=True)
        coverage = np.count_nonzero(prior) / prior.size
        center = prior[sh // 3:2 * sh // 3, sw // 3:2 * sw // 3]
        if coverage < 0.17 or np.count_nonzero(center) < 0.52 * center.size:
            return cv2.resize(bounds, (width, height), interpolation=cv2.INTER_NEAREST), {
                "method": "unresolved", "reason": "No dominant visible paper region", "seed_coverage": coverage,
                **details,
            }
        if coverage == 1:
            return np.full((height, width), 255, np.uint8), {
                "method": "full_visible_frame", "seed_coverage": coverage, **details,
            }
        core = cv2.erode(prior, core_kernel)
        method = "partial_edges" if sides else "visible_region"

    margin = cv2.dilate(prior, kernel)
    labels = np.full(prior.shape, cv2.GC_PR_BGD, np.uint8)
    labels[margin == 0] = cv2.GC_BGD
    labels[prior != 0] = cv2.GC_PR_FGD
    labels[core != 0] = cv2.GC_FGD
    labels[bounds == 0] = cv2.GC_BGD
    if np.any(labels == cv2.GC_FGD) and np.any(labels == cv2.GC_BGD):
        bg = np.zeros((1, 65), np.float64)
        fg = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, labels, None, bg, fg, 2, cv2.GC_INIT_WITH_MASK)
        mask = _component(np.uint8((labels & 1) * 255), center_weight=True)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if np.count_nonzero(mask) < 0.55 * np.count_nonzero(prior):
            mask = prior
            method += "_prior"
    else:
        mask = prior
        method += "_prior"
    mask = cv2.bitwise_and(mask, bounds)
    # Trim resampled desk color at the cut edge without treating the camera frame as background.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=255)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST), {
        "method": method, "visible_fraction": float(np.count_nonzero(mask) / mask.size), **details,
    }


def _print_tone(contrast: np.ndarray, level: np.ndarray, paper: np.ndarray, scale: float) -> np.ndarray:
    # Restore a small optical blur in the captured contrast with a bounded back-projection.
    sigma = 0.61 * scale
    restored = contrast.copy()
    for _ in range(3):
        smoothed = cv2.GaussianBlur(restored, (0, 0), sigma)
        restored += 0.95 * (contrast - smoothed)
        np.clip(restored, 0, 1.08, out=restored)
    detail = restored - cv2.GaussianBlur(restored, (0, 0), sigma)
    restored += 0.30 * np.clip(detail, -0.16, 0.14)

    # Small strokes need a darker mapping. Broad strokes in shadows need more gray levels at their edges.
    stroke_radius = cv2.distanceTransform(np.uint8(contrast > 0.18), cv2.DIST_L2, 5)
    neighborhood = np.ones((2 * round(2 * scale) + 1,) * 2, np.uint8)
    stroke_radius = cv2.dilate(stroke_radius, neighborhood) / scale
    thick = np.clip((stroke_radius - 1.0) / 0.83, 0, 1)
    white_level = float(np.percentile(level[paper], 90))
    shadow = np.clip((0.87 - level / white_level) / 0.29, 0, 1)
    black_point = 0.43 + thick * (0.06 + 0.16 * shadow)

    # Lift low-contrast marks toward white only when they are isolated from stronger ink.
    nearby_ink = cv2.dilate(contrast, np.ones((2 * round(scale) + 1,) * 2, np.uint8))
    white_point = np.maximum(0.036, 3.5 / level) + 0.06 * np.clip((0.18 - nearby_ink) / 0.13, 0, 1)
    ink = np.clip((restored - white_point) / np.maximum(black_point - white_point, 0.13), 0, 1)
    return 255 - 238 * ink * (1 + 0.34 * (1 - ink))


def _white_paper(image: np.ndarray, mask: np.ndarray, settings: dict, *, sampling_scale: float = 1,
                 clean_margin: bool = False) -> np.ndarray:
    paper = mask != 0
    if not np.any(paper):
        return np.full_like(image, 255)
    gray = gray_image(image)
    filled = gray
    if not np.all(paper):
        # Extrapolate the illumination at the paper edge before estimating the background.
        _, nearest = cv2.distanceTransformWithLabels(np.uint8(~paper), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros(int(nearest.max()) + 1, np.uint8)
        lookup[nearest[paper]] = gray[paper]
        filled = lookup[nearest]
        del _, nearest, lookup
    short = min(gray.shape)
    low = max(3, round((settings["background_min"] // 2) * sampling_scale) | 1)
    high = round((settings["background_max"] // 2) * sampling_scale) | 1
    requested = round(short * settings["background_scale"] / 2) | 1
    window = min(short - (short % 2 == 0), max(low, min(high, requested)))
    # The median rejects thin ink and follows cast-shadow edges instead of smearing them.
    background = cv2.medianBlur(filled, window).astype(np.float32)
    level = np.maximum(background, settings["background_floor"])
    gain = np.minimum(255 / level, 3.5)
    gray_normalized = np.minimum(gray * gain, 255)
    chroma = np.ptp(image, axis=2).astype(np.float32)
    chroma /= np.maximum(image.max(axis=2), 1)
    reference = chroma[paper & (gray_normalized >= 245)]
    paper_chroma = float(np.clip(np.percentile(reference, 65) + 0.028, 0.065, 0.17)) if reference.size else 0.08

    stroke_window = 2 * round(3 * sampling_scale) + 1
    stroke = cv2.morphologyEx(filled, cv2.MORPH_BLACKHAT, np.ones((stroke_window, stroke_window), np.uint8))
    colorful = paper & (chroma > paper_chroma + 0.055)
    dark = (paper & ((stroke >= 18) | (gray_normalized < 163))).astype(np.float32)
    density_window = max(9, round(min(gray.shape) * 0.075) | 1)
    dense = np.uint8(cv2.blur(dark, (density_window, density_window)) >= 0.53)
    opening = max(3, round(min(gray.shape) * 0.023) | 1)
    opening_kernel = np.ones((opening, opening), np.uint8)
    dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, opening_kernel) != 0
    color_regions = cv2.morphologyEx(np.uint8(colorful), cv2.MORPH_OPEN, opening_kernel) != 0
    detail = np.uint8((dense & paper) | color_regions)
    span = max(3, round(min(gray.shape) * 0.026) | 1)
    joined = cv2.morphologyEx(detail, cv2.MORPH_CLOSE, np.ones((span, span), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    artwork = np.zeros_like(detail)
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if (w >= 0.10 * gray.shape[1] and h >= 0.07 * gray.shape[0] and
                cv2.contourArea(contour) >= 0.20 * w * h):
            # Keep light parts surrounded by the darker printed texture at their original tone.
            cv2.drawContours(artwork, [cv2.convexHull(contour)], -1, 1, cv2.FILLED)
    rim = max(3, round(short * 0.012) | 1)
    artwork = cv2.dilate(artwork, np.ones((rim, rim), np.uint8))

    contrast = np.maximum((background - gray) / level, 0)
    text_gray = _print_tone(contrast, level, paper, sampling_scale)
    text_gray = np.where(background >= settings["background_floor"], text_gray, gray_normalized)

    color_weight = np.clip((chroma - paper_chroma - 0.025) / 0.08, 0, 1).astype(np.float32)
    colored = np.minimum(image * gain[:, :, None], 255)
    tint = colored - gray_normalized[:, :, None]
    result = np.rint(np.clip(text_gray[:, :, None] + color_weight[:, :, None] * tint, 0, 255)).astype(np.uint8)
    preserve = (artwork != 0) | color_regions | (paper & (chroma > paper_chroma + 0.13))
    result[preserve] = image[preserve]
    result[~paper] = 255
    if clean_margin:
        clean_page_edges(result, paper, artwork)
    return result


def _pad_a4(image: np.ndarray, mask: np.ndarray, orientation: str, missing: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    width, height = _size(image)
    ratio = 210 / 297 if orientation == "portrait" else 297 / 210
    target_w, target_h = max(width, round(height * ratio)), max(height, round(width / ratio))
    dx, dy = target_w - width, target_h - height
    left = dx if missing["left"] and not missing["right"] else 0 if missing["right"] and not missing["left"] else dx // 2
    top = dy if missing["top"] and not missing["bottom"] else 0 if missing["bottom"] and not missing["top"] else dy // 2
    bounds = (top, dy - top, left, dx - left)
    return (cv2.copyMakeBorder(image, *bounds, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
            cv2.copyMakeBorder(mask, *bounds, cv2.BORDER_CONSTANT, value=0),
            {"left": left, "top": top, "right": dx - left, "bottom": dy - top})


def fit_a4_after_dewarp(image: np.ndarray, orientation: str, max_side: int) -> np.ndarray:
    mask = np.full(image.shape[:2], 255, np.uint8)
    page, _, _ = _pad_a4(image, mask, orientation, dict.fromkeys(("left", "top", "right", "bottom"), False))
    return _resize(page, max_side)


def prepare_page(image: np.ndarray, config: dict, page_format: str, orientation: str,
                 manual: np.ndarray | None = None, *, dpi: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    width, height = _size(image)
    if min(width, height) < 3:
        raise DocscanError("page_too_small", "A scan page must have at least three pixels on each side")
    direction = orientation if orientation != "auto" else ("portrait" if height >= width else "landscape")
    detection = detect_page(image, config["geometry"], direction, partial=True) if manual is None else None
    corners = manual if detection is None else detection.corners
    sides = () if detection is None else detection.partial_sides
    mask, segmentation = visible_paper(image, corners, sides, confirm_quad=manual is None)
    if "quad_rejected" in segmentation:
        corners = None
    if cv2.countNonZero(mask) == 0:
        raise DocscanError("empty_paper_mask", "The paper region is empty")
    bound = max(2, round(min(width, height) * 0.004))
    xs, ys, ww, hh = cv2.boundingRect(mask)
    frame_edges = {"top": ys < bound, "right": xs + ww > width - bound,
                   "bottom": ys + hh > height - bound, "left": xs < bound}
    image = image.copy()
    image[mask == 0] = 255
    meta = {"segmentation": segmentation, "frame_edges_touched": frame_edges,
            "needs_review": segmentation["method"] == "unresolved"}
    if detection is not None:
        meta["detection"] = detection.diagnostics
    sampling_scale = 1.0
    if corners is not None:
        if orientation == "auto":
            edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            direction = "portrait" if edges[1] + edges[3] >= edges[0] + edges[2] else "landscape"
        aspect = projected_page_aspect(corners, (width, height)) if page_format == "native" else None
        if aspect is not None:
            meta["estimated_page_aspect"] = aspect
            if orientation == "auto":
                direction = "landscape" if aspect >= 1 else "portrait"
        settings = {**config["geometry"], "page_format": page_format}
        nominal = rectified_size(corners, settings, direction, aspect=aspect)
        page_size, target = pdf_raster_layout(nominal, page_format, direction, config["pdf"]["dpi"], dpi)
        rectified, geometry = warp_document(image, corners, settings, direction, interpolation="cubic", output_size=target)
        mask = cv2.warpPerspective(mask, np.asarray(geometry["input_to_rectified"]), tuple(geometry["output_size"]),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if dpi is not None:
            optical_settings = {**settings, "max_long_edge": 2 * max(width, height)}
            optical = rectified_size(corners, optical_settings, direction, aspect=aspect)
            sampling_scale = min(target[0] / max(1, optical[0]), target[1] / max(1, optical[1]))
        meta.update(mode="manual_edges" if manual is not None else "rectified_edges", geometry=geometry)
    else:
        if orientation == "auto":
            direction = "portrait" if hh >= ww else "landscape"
        rectified = image[ys:ys + hh, xs:xs + ww]
        mask = mask[ys:ys + hh, xs:xs + ww]
        meta.update(mode="visible_only", crop=[xs, ys, ww, hh])
        if page_format == "a4":
            rectified, mask, pad = _pad_a4(rectified, mask, direction, frame_edges)
            meta["white_padding_pixels"] = pad
        optical = _size(rectified)
        nominal = _limited_size(optical, config["geometry"]["max_long_edge"])
        page_size, target = pdf_raster_layout(nominal, page_format, direction, config["pdf"]["dpi"], dpi)
        rectified = resize_page(rectified, target)
        mask = cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST)
        if dpi is not None:
            sampling_scale = min(target[0] / optical[0], target[1] / optical[1])
    if page_size is not None:
        meta["pdf_page_size_points"] = page_size
    scale = float(np.clip(sampling_scale, 1, 3))
    clean_margin = corners is not None and not any(frame_edges.values())
    return (_white_paper(rectified, mask, config["enhance"], sampling_scale=scale, clean_margin=clean_margin),
            mask, {**meta, "orientation": direction})
