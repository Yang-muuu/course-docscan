from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from .common import DocscanError, elapsed_ms


def _gray(gray: np.ndarray) -> None:
    if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 3:
        raise DocscanError("invalid_gray", "Expected a grayscale uint8 image with both sides at least 3 pixels")


def gray_image(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def correct_light(gray: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, int]:
    _gray(gray)
    short_side = min(gray.shape)
    largest_odd = short_side - (short_side % 2 == 0)
    requested = 2 * int(config["background_scale"] * short_side / 2) + 1
    size = min(largest_odd, max(config["background_min"], min(config["background_max"], requested)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    background = cv2.GaussianBlur(background, (size, size), sigmaX=max(0.5, size / 8), borderType=cv2.BORDER_REPLICATE)
    denominator = np.maximum(background.astype(np.float32), config["background_floor"])
    corrected = np.clip(np.rint(config["target_white"] * gray.astype(np.float32) / denominator), 0, 255)
    return background, corrected.astype(np.uint8), size


def otsu(gray: np.ndarray) -> tuple[np.ndarray, float]:
    _gray(gray)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary, threshold


def adaptive(gray: np.ndarray, config: dict) -> tuple[np.ndarray, int]:
    _gray(gray)
    requested = config["adaptive_block"]
    if type(requested) is not int or requested < 3 or requested % 2 != 1:
        raise DocscanError("invalid_adaptive_block", "Adaptive window must be an odd integer at least 3")
    short_side = min(gray.shape)
    block = min(requested, short_side - (short_side % 2 == 0))
    result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   block, config["adaptive_c"])
    return result, block


def make_variants(gray: np.ndarray, config: dict) -> tuple[dict, np.ndarray, dict, dict]:
    variants = {"B": gray}
    times = {"B": 0.0}
    start = perf_counter()
    variants["C"], threshold = otsu(gray)
    times["C"] = elapsed_ms(start)
    start = perf_counter()
    variants["D"], block = adaptive(gray, config)
    times["D"] = elapsed_ms(start)
    start = perf_counter()
    background, variants["E"], kernel = correct_light(gray, config)
    times["E"] = elapsed_ms(start)
    start = perf_counter()
    variants["F"], _ = adaptive(variants["E"], config)
    times["F"] = round(times["E"] + elapsed_ms(start), 3)
    return variants, background, times, {"otsu_threshold": threshold, "background_kernel": kernel, "adaptive_block": block}
