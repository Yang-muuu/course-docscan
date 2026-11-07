from __future__ import annotations

import cv2
import numpy as np

from .common import DocscanError
from .enhance import gray_image
from .page_margin import _printed_rules


class CaptureMismatch(DocscanError):
    def __init__(self, capture: int, detail: str, evidence: dict | None = None):
        super().__init__("capture_mismatch", f"Capture {capture} does not reliably match the same page: {detail}")
        self.evidence = {"capture": capture, "reason": detail, **(evidence or {})}


def _local_ink_differences(reference: np.ndarray, candidate: np.ndarray, *,
                           pixel_fraction: float = 0.016, local_fraction: float = 0.14) -> list[dict]:
    """Find concentrated missing strokes, ignoring faint edges and ruled lines."""
    height, width = reference.shape
    short = min(height, width)
    yy, xx = np.ogrid[:height, :width]
    dx, dy = np.minimum(xx, width - 1 - xx), np.minimum(yy, height - 1 - yy)
    depth = np.minimum(dx, dy)
    possible_ink = (reference < 201) | (candidate < 201)
    rules = _printed_rules(possible_ink.astype(np.uint8), depth.astype(np.float32), dx, dy, depth, short)
    interior = (depth >= max(2, round(0.024 * short))) & (cv2.dilate(rules, np.ones((5, 5), np.uint8)) == 0)
    kernel = np.ones((3, 3), np.uint8)
    minimum = max(6, round(pixel_fraction * short))
    padding = max(4, round(0.007 * short))
    differences = []
    for side, dark, other in (("reference", reference, candidate), ("capture", candidate, reference)):
        near_other = cv2.dilate(np.uint8(other < 201), kernel)
        missing = (dark < 160) & (near_other == 0) & interior
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(missing.astype(np.uint8), kernel))
        for label, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if min(w, h) < 4 or max(w, h) > 6 * min(w, h):
                continue
            region = np.s_[y:y + h, x:x + w]
            count = int(np.count_nonzero(missing[region] & (labels[region] == label)))
            if count < minimum:
                continue
            context = np.s_[max(0, y - padding):y + h + padding, max(0, x - padding):x + w + padding]
            fraction = count / max(1, np.count_nonzero(possible_ink[context]))
            if fraction >= local_fraction:
                differences.append({"side": side, "box": [int(x), int(y), int(w), int(h)],
                                    "missing_ink_pixels": count, "local_fraction": round(fraction, 4)})
    return sorted(differences, key=lambda item: item["missing_ink_pixels"], reverse=True)


def _rigid_ink_conflicts(reference: np.ndarray, warped: np.ndarray, shift: np.ndarray, regions: list[dict]) -> list[dict]:
    """Check glyph shape without letting local optical flow deform the glyph itself."""
    height, width = reference.shape
    short = min(height, width)
    padding = max(4, round(0.007 * short))
    radius = max(2, round(0.003 * short))
    conflicts = []
    for region in regions:
        x, y, w, h = region["box"]
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
        roi = np.s_[y0:y1, x0:x1]
        glyph = reference[roi]
        patch_h, patch_w = glyph.shape
        dx, dy = np.median(shift[roi].reshape(-1, 2), axis=0)
        center = (float(np.clip((x0 + x1 - 1) / 2 + dx, 0, width - 1)),
                  float(np.clip((y0 + y1 - 1) / 2 + dy, 0, height - 1)))
        search = cv2.getRectSubPix(warped, (patch_w + 2 * radius, patch_h + 2 * radius), center)
        scores = cv2.matchTemplate(search, glyph, cv2.TM_CCOEFF_NORMED)
        _, similarity, _, (best_x, best_y) = cv2.minMaxLoc(scores)
        counterpart = search[best_y:best_y + patch_h, best_x:best_x + patch_w]
        ink = min(np.count_nonzero(glyph < 180), np.count_nonzero(counterpart < 180))
        if ink >= max(6, round(0.009 * short)) and similarity < 0.88:
            conflicts.append({**region, "rigid_similarity": round(similarity, 4)})
    return conflicts


def _register(reference: np.ndarray, candidate: np.ndarray, detector, keypoints, descriptors,
              capture: int) -> tuple[np.ndarray, np.ndarray, dict]:
    def mismatch(detail: str) -> DocscanError:
        return CaptureMismatch(capture, detail)

    points, features = detector.detectAndCompute(candidate, None)
    if features is None or descriptors is None or min(len(features), len(descriptors)) < 40:
        raise mismatch("too few printed landmarks")
    pairs = cv2.BFMatcher().knnMatch(features, descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.74 * second.distance]
    if len(matches) < 40:
        raise mismatch("too few matching printed landmarks")

    source = np.float32([points[match.queryIdx].pt for match in matches])
    target = np.float32([keypoints[match.trainIdx].pt for match in matches])
    matrix, inliers = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 2.5)
    height, width = reference.shape
    short = min(height, width)
    if matrix is None or inliers is None or int(inliers.sum()) < 40 or inliers.mean() < 0.48:
        raise mismatch("unstable perspective alignment")
    coverage = cv2.contourArea(cv2.convexHull(target[inliers.ravel() != 0])) / (width * height)
    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    corner_shift = cv2.perspectiveTransform(corners, matrix) - corners
    if coverage < 0.18 or not np.isfinite(corner_shift).all() or np.max(np.abs(corner_shift)) > 0.16 * short:
        raise mismatch("matches do not cover the page")

    warped = cv2.warpPerspective(candidate, matrix, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(0)
    shift = flow.calc(reference, warped, None)
    kernel = np.ones((5, 5), np.uint8)
    reference_ink = reference < 180
    near_reference = cv2.dilate(reference_ink.astype(np.uint8), kernel) != 0
    magnitude = np.linalg.norm(shift, axis=2)
    if not np.any(near_reference) or np.percentile(magnitude[near_reference], 90) > 0.012 * short:
        raise mismatch("printed details move too far after perspective alignment")
    shift *= np.minimum(1, 0.029 * short / np.maximum(magnitude, 1e-6))[:, :, None]
    yy, xx = np.ogrid[:height, :width]
    aligned = cv2.remap(warped, xx.astype(np.float32) + shift[:, :, 0], yy.astype(np.float32) + shift[:, :, 1],
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ink = aligned < 180
    near_ink = cv2.dilate(ink.astype(np.uint8), kernel) != 0
    # Verify the printed content as well as the shared page layout.
    support = reference_ink.astype(np.uint8) + ink
    matched = (reference_ink & near_ink).astype(np.uint8) + (ink & near_reference)
    agreement = float(matched.sum() / max(1, support.sum()))
    before = _local_ink_differences(reference, warped, pixel_fraction=0.008, local_fraction=0.085)
    evidence = {"capture": capture, "matches": int(inliers.sum()), "ink_agreement": round(agreement, 4),
                "alignment_size": [width, height], "before_flow": before[:12],
                "after_flow": _local_ink_differences(reference, aligned)[:12],
                "rigid_conflicts": _rigid_ink_conflicts(reference, warped, shift, before)[:12]}
    if evidence["after_flow"]:
        raise CaptureMismatch(capture, "a local printed detail differs after alignment", evidence)
    if evidence["rigid_conflicts"]:
        raise CaptureMismatch(capture, "a local glyph shape differs under translation", evidence)
    tile = max(32, short // 6)
    # Fold and rim differences count toward the overall check, not the local text veto.
    margin = round(0.055 * short)
    for y in range(margin, height - margin, tile):
        for x in range(margin, width - margin, tile):
            roi = np.s_[y:min(y + tile, height - margin), x:min(x + tile, width - margin)]
            count = int(support[roi].sum())
            if count >= 140 and matched[roi].sum() < 0.81 * count:
                evidence["conflict_tile"] = [x, y, min(tile, width - margin - x), min(tile, height - margin - y)]
                raise CaptureMismatch(capture, "printed content differs in a local area", evidence)
    if agreement < 0.955:
        raise CaptureMismatch(capture, "printed content does not agree", evidence)
    return np.linalg.inv(matrix), shift, evidence


def _resample(image: np.ndarray, size: tuple[int, int], inverse: np.ndarray, shift: np.ndarray) -> np.ndarray:
    height, width = size
    sh, sw = shift.shape[:2]
    grid = cv2.resize(shift, (width, height), interpolation=cv2.INTER_LINEAR)
    grid[:, :, 0] += (np.arange(width, dtype=np.float32) + 0.5) * (sw / width) - 0.5
    grid[:, :, 1] += ((np.arange(height, dtype=np.float32) + 0.5) * (sh / height) - 0.5)[:, None]
    # Sample the full-resolution capture once, composing its perspective and local alignment.
    positions = cv2.perspectiveTransform(grid, inverse)
    positions[:, :, 0] = (positions[:, :, 0] + 0.5) * (image.shape[1] / sw) - 0.5
    positions[:, :, 1] = (positions[:, :, 1] + 0.5) * (image.shape[0] / sh) - 0.5
    return cv2.remap(image, positions, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255, 255, 255))


def _preserve_margin_rules(result: np.ndarray, reference: np.ndarray, gray: np.ndarray) -> None:
    height, width = gray.shape
    short = min(height, width)
    yy, xx = np.ogrid[:height, :width]
    dx, dy = np.minimum(xx, width - 1 - xx), np.minimum(yy, height - 1 - yy)
    depth = np.minimum(dx, dy)
    ink = np.uint8(gray < 236)
    rules = _printed_rules(ink, depth.astype(np.float32), dx, dy, depth, short)
    if not np.any(rules):
        return

    length = max(13, round(0.021 * short))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    supported = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    band = cv2.dilate(rules, np.ones((9, 9), np.uint8))
    full_size = reference.shape[1], reference.shape[0]
    band = cv2.resize(band, full_size, interpolation=cv2.INTER_NEAREST) != 0
    supported = cv2.resize(supported, full_size, interpolation=cv2.INTER_NEAREST) != 0
    # Keep supported ink dark and its surrounding paper light, without restoring a crossing scratch.
    reference_gray, result_gray = gray_image(reference), gray_image(result)
    restore_print = supported & (reference_gray < result_gray)
    restore_paper = ~supported & (reference_gray > result_gray)
    restore = band & (restore_print | restore_paper)
    result[restore] = reference[restore]


def fuse_captures(images: list[np.ndarray]) -> tuple[np.ndarray, int, list[dict]]:
    """Combine registered photographs of one page while keeping the anchor raster size."""
    anchor = max(range(len(images)), key=lambda index: images[index].shape[0] * images[index].shape[1])
    reference = images[anchor]
    height, width = reference.shape[:2]
    scale = min(1, 1800 / max(height, width))
    work_size = round(width * scale), round(height * scale)
    gray = cv2.resize(gray_image(reference), work_size, interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.016)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    aligned = [reference]
    matches = []
    for index, image in enumerate(images):
        if index == anchor:
            continue
        candidate = cv2.resize(gray_image(image), work_size, interpolation=cv2.INTER_AREA)
        try:
            inverse, shift, report = _register(gray, candidate, detector, keypoints, descriptors, index + 1)
        except CaptureMismatch as error:
            error.evidence.update(anchor_capture=anchor + 1, registration=matches)
            raise
        aligned.append(_resample(image, (height, width), inverse, shift))
        matches.append(report)

    result = np.empty_like(reference)
    for y in range(0, height, 256):
        rows = slice(y, y + 256)
        values = np.stack([image[rows] for image in aligned])
        average = values.mean(axis=0, dtype=np.float32)
        if len(aligned) == 2:
            result[rows] = np.rint(average).astype(np.uint8)
            continue
        order = np.argsort(np.stack([gray_image(image[rows]) for image in aligned]), axis=0)
        rank = order[(len(aligned) - 1) // 2]
        consensus = np.take_along_axis(values, rank[None, :, :, None], axis=0)[0]
        blend = 0.84 * consensus + 0.16 * average
        result[rows] = np.rint(np.where(consensus.min(axis=2, keepdims=True) >= 251, consensus, blend)).astype(np.uint8)
    _preserve_margin_rules(result, reference, gray)
    return result, anchor, matches
