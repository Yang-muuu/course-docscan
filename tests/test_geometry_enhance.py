import json

import cv2
import numpy as np
import pytest
from PIL import Image

from docscan.common import DocscanError
from docscan.config import DEFAULTS, load_config
from docscan.enhance import adaptive, correct_light, otsu
from docscan.geometry import CORNER_ORDER, RAW_SPACE, detect_page, load_annotation, validate_quad, warp_document
from docscan.image_io import NORMALIZED_SPACE, load_photo


def test_known_projection_rectifies_without_clipping(samples):
    photo = load_photo(samples / "sample.jpg")
    marked = load_annotation(samples / "sample.corners.json", photo)
    image, geometry = warp_document(photo.bgr, marked.corners, DEFAULTS["geometry"], "portrait")
    w, h = geometry["output_size"]
    assert image.shape == (h, w, 3)
    assert w / h == pytest.approx(210 / 297, abs=0.002)
    assert np.allclose(np.asarray(geometry["input_to_rectified"]) @ geometry["rectified_to_input"], np.eye(3))
    projected = cv2.perspectiveTransform(marked.corners[None], np.asarray(geometry["input_to_rectified"]))[0]
    assert np.allclose(projected, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], atol=0.002)
    text = cv2.perspectiveTransform(np.float32(marked.text_polygons), np.asarray(geometry["input_to_rectified"]))
    assert np.all(text > 0) and np.all(text < [w, h])
    for polygon in text:
        x, y, ww, hh = cv2.boundingRect(polygon)
        assert image[y:y + hh, x:x + ww].std() > 2


@pytest.mark.parametrize("bad", [
    [[8, 8], [90, 8], [90, 90], [90, 90]],
    [[8, 8], [35, 35], [60, 60], [90, 90]],
    [[8, 8], [90, 90], [90, 8], [8, 90]],
    [[90, 8], [90, 90], [8, 90], [8, 8]],
    [[-1, 8], [90, 8], [90, 90], [8, 90]],
])
def test_bad_quad_rejected(bad):
    with pytest.raises(DocscanError, match="Corner|corner"):
        validate_quad(bad, (100, 100))


def test_unicode_exif_and_raw_corners(tmp_path):
    filename = tmp_path / "纸张旋转.jpg"
    img = Image.new("RGB", (80, 120), (240, 240, 240))
    exif = img.getexif()
    exif[274] = 6
    img.save(filename, exif=exif)
    photo = load_photo(filename)
    assert photo.raw_size == (80, 120) and photo.size == (120, 80)
    data = {"coordinate_space": RAW_SPACE, "image_size": [80, 120], "order": CORNER_ORDER,
            "points": [[8, 12], [70, 12], [70, 108], [8, 108]]}
    corners = tmp_path / "纸张角点.json"
    corners.write_text(json.dumps(data), encoding="utf-8")
    assert np.array_equal(load_annotation(corners, photo).corners, [[11, 8], [107, 8], [107, 70], [11, 70]])
    corners.write_text(json.dumps({**data, "coordinate_space": NORMALIZED_SPACE}), encoding="utf-8")
    with pytest.raises(DocscanError, match="does not match"):
        load_annotation(corners, photo)


def test_detection_scales_back_and_blank_fails(samples):
    photo = load_photo(samples / "sample.jpg")
    annotation = load_annotation(samples / "sample.corners.json", photo)
    detection = detect_page(photo.bgr, DEFAULTS["geometry"], "portrait")
    assert detection.corners is not None, detection.diagnostics
    assert max(detection.diagnostics["detection_size"]) < max(photo.size)
    assert np.linalg.norm(detection.corners - annotation.corners, axis=1).max() < 18
    blank = detect_page(np.full((550, 400, 3), 200, np.uint8), DEFAULTS["geometry"], "portrait")
    assert blank.corners is None and blank.diagnostics["reason"]


@pytest.mark.parametrize("base", [0, 255, None])
def test_enhancement_numerics_and_thresholding(base):
    gradient = np.tile(np.linspace(34, 218, 97, dtype=np.uint8), (76, 1))
    gray = np.full(gradient.shape, base, dtype=np.uint8) if base is not None else gradient
    config = DEFAULTS["enhance"]
    background, enhanced, scale = correct_light(gray, config)
    assert scale % 2 == 1
    for result in (background, enhanced, otsu(gray)[0], adaptive(gray, config)[0]):
        assert result.shape == gray.shape and result.dtype == np.uint8 and np.isfinite(result).all()
    if base is None:
        assert enhanced[:, 8:-8].std() < gray[:, 8:-8].std()
    for result in (otsu(gray)[0], adaptive(enhanced, config)[0]):
        assert set(np.unique(result)) <= {0, 255}


def test_illegal_window_and_config(tmp_path):
    bad = {**DEFAULTS["enhance"], "adaptive_block": 6}
    with pytest.raises(DocscanError, match="odd"):
        adaptive(np.zeros((8, 8), dtype=np.uint8), bad)
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("enhance:\n  adaptive_block: 6\n", encoding="utf-8")
    with pytest.raises(DocscanError, match="adaptive_block"):
        load_config(config_path)
