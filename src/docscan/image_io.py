from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .common import DocscanError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NORMALIZED_SPACE = "exif_normalized_pixels"
WARPED_SPACE = "rectified_pixels"


@dataclass(frozen=True)
class Photo:
    bgr: np.ndarray
    raw_size: tuple[int, int]
    exif_orientation: int
    sha256: str

    @property
    def size(self) -> tuple[int, int]:
        return self.bgr.shape[1], self.bgr.shape[0]


def load_photo(path: Path) -> Photo:
    try:
        data = path.read_bytes()
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise DocscanError("unsupported_image", f"Expected JPG or PNG: {path}")
            orientation = image.getexif().get(274, 1)
            orientation = orientation if type(orientation) is int and 1 <= orientation <= 8 else 1
            upright = ImageOps.exif_transpose(image)
            if "A" in upright.getbands() or "transparency" in upright.info:
                white = Image.new("RGBA", upright.size, "white")
                upright = Image.alpha_composite(white, upright.convert("RGBA"))
            pixels = cv2.cvtColor(np.asarray(upright.convert("RGB")), cv2.COLOR_RGB2BGR)
            return Photo(pixels, image.size, orientation, hashlib.sha256(data).hexdigest())
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raise DocscanError("image_read_failed", f"Cannot decode {path}: {error}") from error


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise DocscanError("image_write_failed", "PNG encoding failed")
    return encoded.tobytes()


def write_png(path: Path, image: np.ndarray) -> None:
    path.write_bytes(encode_png(image))


def raw_to_upright(points: np.ndarray, photo: Photo) -> np.ndarray:
    x, y = points.T
    w, h = photo.raw_size
    conversions = {
        1: (x, y), 2: (w - 1 - x, y), 3: (w - 1 - x, h - 1 - y), 4: (x, h - 1 - y),
        5: (y, x), 6: (h - 1 - y, x), 7: (h - 1 - y, w - 1 - x), 8: (y, w - 1 - x),
    }
    return np.column_stack(conversions[photo.exif_orientation])

