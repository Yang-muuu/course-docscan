import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from docscan.capture_fusion import _preserve_margin_rules
from docscan.cli import main
from docscan.common import dependency_versions
from docscan.config import DEFAULTS
from docscan.fonts import chinese_font
from docscan.geometry import detect_page, projected_page_aspect
from docscan.image_io import load_photo, write_png
from docscan.models import model_status
from docscan.page_margin import clean_page_edges
from docscan.page_surface import _white_paper, prepare_page
from docscan.samples import DOCUMENTS, render_page
from docscan.spread import find_gutter


DESK = (63, 116, 154)
ART = (180, 91, 44)


def _damaged_page() -> np.ndarray:
    image = np.full((820, 680, 3), DESK, dtype=np.uint8)
    paper = np.full((660, 470, 3), (219, 230, 234), dtype=np.uint8)
    for y in range(72, 615, 40):
        cv2.putText(paper, "SYNTHETIC PRINT", (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    paper[255:410, 130:375] = ART
    paper[150:164, 95:125] = (65, 173, 67)
    paper[572:596, 145:190] = (47, 119, 232)
    image[68:728, 105:575] = paper
    cv2.fillConvexPoly(image, np.int32([[105, 68], [232, 68], [105, 202]]), DESK)
    return image


def _read_run(path: Path) -> dict:
    return json.loads((path / "run.json").read_text(encoding="utf-8"))


def test_infers_all_four_corners_from_interrupted_edges():
    scene = np.full((710, 580, 3), DESK, dtype=np.uint8)
    expected = np.int32([[78, 60], [482, 70], [500, 656], [71, 641]])
    cv2.fillConvexPoly(scene, expected, (231, 237, 240))
    for index, corner in enumerate(expected):
        prev, following = expected[(index - 1) % 4], expected[(index + 1) % 4]
        hide = np.array([corner, corner + 0.16 * (prev - corner), corner + 0.16 * (following - corner)])
        cv2.fillConvexPoly(scene, np.rint(hide).astype(np.int32), DESK)
    result = detect_page(scene, DEFAULTS["geometry"], "portrait")
    assert result.corners is not None, result.diagnostics
    assert result.diagnostics["candidate_source"] == "fitted_lines"
    assert np.linalg.norm(result.corners - expected, axis=1).max() < 8


def test_native_aspect_recovers_a_tilted_rectangle():
    aspect = 297 / 210
    corners = np.array([[0, 0, 0], [aspect, 0, 0], [aspect, 1, 0], [0, 1, 0]])
    rotation = cv2.Rodrigues(np.array([0.41, -0.30, 0.12]))[0]
    camera = corners @ rotation.T + [-0.75, -0.42, 3.2]
    projected = 900 * camera[:, :2] / camera[:, 2:] + [639.5, 479.5]
    assert projected_page_aspect(projected, (1280, 960)) == pytest.approx(aspect, abs=0.001)
    front = np.array([[80, 100], [680, 100], [680, 500], [80, 500]])
    assert projected_page_aspect(front, (1280, 960)) is None


def test_torn_a4_pdf_is_white_and_preserves_art(tmp_path):
    photo, output = tmp_path / "缺角纸张.png", tmp_path / "扫描"
    write_png(photo, _damaged_page())
    assert main(["to-pdf", "--input", str(photo), "--output", str(output), "--page-format", "a4",
                 "--source", "synthetic"]) == 0
    run = _read_run(output)
    entry, = run["pages"]
    assert entry["mode"] == "rectified_edges" and run["source"] == "synthetic"
    assert entry["geometry"]["interpolation"] == "cubic"
    assert entry["size"] == [2480, 3508]
    assert run["pdf"]["page_exports"][0]["effective_dpi"] == pytest.approx([300, 300], abs=0.05)
    raster = load_photo(output / entry["image"]).bgr
    mask = load_photo(output / entry["mask"]).bgr[:, :, 0]
    assert np.count_nonzero(mask == 0) > 1000
    assert np.all(raster[18:45, 18:45] == 255) and np.all(raster[mask == 0] == 255)
    assert np.all(raster[:, -1] == 255)
    assert np.count_nonzero(np.all(raster == ART, axis=-1)) > 24000
    with pymupdf.open(output / "book.pdf") as document:
        assert len(document) == 1
        page = document[0]
        assert page.rect.width == pytest.approx(210 * 72 / 25.4, abs=0.05)
        assert page.rect.height == pytest.approx(297 * 72 / 25.4, abs=0.05)
        pix = pymupdf.Pixmap(document, page.get_images()[0][0])
        stored = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        assert np.array_equal(stored, cv2.cvtColor(raster, cv2.COLOR_BGR2RGB))


def test_dpi_changes_native_image_pixels_without_changing_page_size(tmp_path):
    photo = tmp_path / "paper.png"
    write_png(photo, np.full((150, 210, 3), 235, dtype=np.uint8))
    rects, pixels = [], []
    for dpi in (150, 300):
        output = tmp_path / str(dpi)
        assert main(["to-pdf", "--input", str(photo), "--output", str(output), "--dpi", str(dpi)]) == 0
        with pymupdf.open(output / "book.pdf") as document:
            page = document[0]
            rects.append(tuple(page.rect))
            pixels.append(page.get_images()[0][2:4])
    assert rects[0] == rects[1]
    assert pixels == [(210, 150), (420, 300)]


def test_clipped_page_keeps_marks_at_both_frame_edges():
    photo = _damaged_page()[200:670, 48:620]
    page, mask, info = prepare_page(photo, DEFAULTS, "a4", "portrait")
    assert info["mode"] == "visible_only"
    assert info["frame_edges_touched"]["top"] and info["frame_edges_touched"]["bottom"]
    assert info["white_padding_pixels"]["top"] > 30 and info["white_padding_pixels"]["bottom"] > 30
    assert np.all(page[mask == 0] == 255)
    assert np.count_nonzero(np.all(page == (65, 173, 67), axis=-1)) > 100
    assert np.count_nonzero(np.all(page == (47, 119, 232), axis=-1)) > 400


@pytest.mark.parametrize("penumbra", [1, 18])
def test_white_page_keeps_fine_text_across_a_shadow(penumbra):
    clean = np.full((460, 640), 238, dtype=np.uint8)
    for y, words in ((90, "Synthetic pale print"), (170, "Sum 3x + 1 = 16"), (245, "Tiny 42")):
        cv2.putText(clean, words, (115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, 188, 1)
    cv2.line(clean, (140, 286), (505, 286), 176, 1)
    yy, xx = np.indices(clean.shape)
    edge = xx + 0.47 * yy - 440
    light = 0.33 + 0.61 * (0.5 + 0.5 * np.tanh(edge / penumbra))
    noise = np.random.default_rng(4).normal(0, 0.75, clean.shape)
    photo = np.rint(clean * light + noise).astype(np.uint8)
    result = _white_paper(cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR), np.full_like(clean, 255), DEFAULTS["enhance"])[:, :, 0]
    blank = cv2.erode(np.uint8(clean == 238), np.ones((5, 5), np.uint8)) != 0
    assert np.percentile(result[blank & (np.abs(edge) < 20)], 5) >= 248
    assert np.percentile(result[(clean == 188) & (light < 0.5)], 90) <= 125
    assert np.count_nonzero(result[286, 155:295] < 100) >= 135
    assert np.count_nonzero(result[286, 275:336] < 120) >= 58


def test_gray_illustration_keeps_light_areas_inside_dense_print():
    image = np.full((380, 500, 3), 236, dtype=np.uint8)
    yy, xx = np.indices((196, 286))
    texture = np.where((xx % 6 == 0) & (yy % 5 < 3), 194, 55).astype(np.uint8)
    image[100:296, 100:386] = texture[:, :, None]
    image[154:231, 242:352] = 171
    result = _white_paper(image, np.full(image.shape[:2], 255, np.uint8), DEFAULTS["enhance"])
    assert np.array_equal(result[170:218, 265:330], image[170:218, 265:330])
    assert np.all(result[18:66, 30:450] == 255)


def test_blurred_small_strokes_keep_their_one_pixel_gap():
    paper = np.full((450, 590), 235, np.uint8)
    paper[115:123, 126:128] = 22
    paper[115:123, 129:131] = 22
    blurred = cv2.GaussianBlur(paper.astype(np.float32), (5, 5), 0.8)
    photo = np.rint(blurred * 0.42).astype(np.uint8)
    result = _white_paper(cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR), np.full_like(paper, 255), DEFAULTS["enhance"])[:, :, 0]
    left, gap, right = result[119, [127, 128, 129]].astype(int)
    assert max(left, right) <= 50
    assert gap >= max(left, right) + 95
    assert np.all(result[90:100, 110:144] == 255)


def test_edge_cleanup_keeps_the_ruled_line_joining_a_corner_fold():
    image = np.full((620, 860, 3), 255, np.uint8)
    paper = np.ones(image.shape[:2], np.uint8)
    artwork = np.zeros_like(paper)
    cv2.line(image, (18, 26), (810, 26), (25, 25, 25), 2)
    cv2.polylines(image, [np.int32([[5, 0], [14, 18], [25, 28], [15, 48]])], False, (40, 40, 40), 2)
    cv2.line(image, (60, 618), (818, 618), (60, 60, 60), 3)
    cv2.putText(image, "37", (26, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    image[190:264, :44] = ART
    artwork[190:264, :44] = 1
    result = clean_page_edges(image.copy(), paper, artwork)
    assert np.all(result[1:17, :25] == 255)
    assert np.all(result[615:, 55:825] == 255)
    assert np.array_equal(result[24:29, 55:810], image[24:29, 55:810])
    assert np.array_equal(result[64:84, 22:55], image[64:84, 22:55])
    assert np.array_equal(result[190:264, :44], image[190:264, :44])


def test_auto_gutter_splits_two_pages_and_keeps_uncertain_spread(tmp_path):
    spread = np.full((450, 700, 3), (226, 232, 238), dtype=np.uint8)
    for x in (40, 410):
        for y in range(65, 400, 43):
            cv2.putText(spread, "SYNTHETIC", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (28, 28, 28), 2)
    cv2.line(spread, (338, 0), (358, 449), (74, 75, 80), 13)
    line, _ = find_gutter(spread)
    assert line is not None and abs(line[0] - 338 / 699) < 0.035 and abs(line[1] - 358 / 699) < 0.035
    inputs = tmp_path / "photos"
    inputs.mkdir()
    write_png(inputs / "2.png", spread)
    write_png(inputs / "10.png", np.full_like(spread, 231))
    output = tmp_path / "book"
    assert main(["book", "--layout", "spread", "--input", str(inputs), "--output", str(output), "--source", "synthetic"]) == 0
    run = _read_run(output)
    assert [Path(image["input"]).name for image in run["images"]] == ["2.png", "10.png"]
    assert [page["side"] for page in run["pages"]] == ["left", "right", "unsplit"]
    assert run["review_pages"] == [3]
    with pymupdf.open(output / "book.pdf") as document:
        assert len(document) == 3


def test_damaged_file_does_not_publish_an_incomplete_book(tmp_path):
    inputs = tmp_path / "photos"
    inputs.mkdir()
    (inputs / "1.jpg").write_bytes(b"not a photo")
    write_png(inputs / "2.png", _damaged_page())
    output = tmp_path / "book"
    assert main(["to-pdf", "--input", str(inputs), "--output", str(output)]) == 1
    run = _read_run(output)
    assert run["status"] == "incomplete" and [row["status"] for row in run["images"]] == ["error", "ok"]
    assert run["pages"][0]["status"] == "ok"
    assert not (output / "book.pdf").exists()


@pytest.mark.parametrize("different", [False, True])
def test_same_page_fusion_checks_printed_content(tmp_path, different):
    font = chinese_font()
    page = render_page(DOCUMENTS[0], font)[0]
    second = page.copy()
    if different:
        other = render_page(DOCUMENTS[1], font)[0]
        second[240:620] = other[240:620]
    else:
        second[260:] = np.rint(second[260:] * 0.82).astype(np.uint8)
    inputs = tmp_path / "photos"
    inputs.mkdir()
    first = cv2.copyMakeBorder(page, 72, 72, 72, 72, cv2.BORDER_CONSTANT, value=DESK)
    second = cv2.copyMakeBorder(second, 72, 72, 72, 72, cv2.BORDER_CONSTANT, value=DESK)
    height, width = second.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), 2.4, 0.98)
    write_png(inputs / "1.png", first)
    write_png(inputs / "2.png", cv2.warpAffine(second, rotation, (width, height), borderValue=DESK))

    output = tmp_path / "book"
    code = main(["to-pdf", "--input", str(inputs), "--output", str(output), "--same-page", "--dpi", "150"])
    run = _read_run(output)
    if different:
        assert code == 1 and run["error_code"] == "capture_mismatch"
        assert not (output / "book.pdf").exists()
    else:
        assert code == 0 and len(run["capture_pages"]) == 2
        assert run["pages"][0]["mode"] == "same_page"
        with pymupdf.open(output / "book.pdf") as document:
            assert len(document) == 1
            assert document[0].get_images()[0][2:4] == tuple(run["pages"][0]["size"])


def test_fusion_preserves_a_continuous_rule_near_the_margin():
    anchor = np.full((620, 860, 3), 255, np.uint8)
    cv2.line(anchor, (38, 26), (810, 26), (55, 55, 55), 1)
    fused = anchor.copy()
    cv2.line(anchor, (185, 21), (185, 32), (50, 50, 50), 1)
    fused[26, 200:580] = 255
    cv2.line(fused, (200, 30), (579, 30), (110, 110, 110), 1)
    _preserve_margin_rules(fused, anchor, cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY))
    assert np.all(fused[26, 200:580] == 55)
    assert np.all(fused[30, 200:580] == 255)
    assert np.all(fused[31:33, 185] == 255)


@pytest.mark.ocr_integration
def test_searchable_book_keeps_chinese_reading_order(tmp_path):
    versions = dependency_versions()
    if any(not versions[p] for p in ("paddlepaddle", "paddleocr", "paddlex")) or not all(
        entry["ready"] for entry in model_status(Path(DEFAULTS["ocr"]["model_dir"])).values()
    ):
        pytest.skip("Install the pinned PaddleOCR dependencies and weights")
    font = chinese_font()
    left = render_page(DOCUMENTS[0], font)[0]
    right = render_page(DOCUMENTS[1], font)[0]
    seam = np.full((left.shape[0], 19, 3), 130, np.uint8)
    photo = tmp_path / "摊开.png"
    write_png(photo, np.concatenate((left, seam, right), axis=1))
    output = tmp_path / "book"
    assert main(["to-pdf", "--input", str(photo), "--layout", "spread", "--pdf-mode", "searchable",
                 "--output", str(output), "--source", "synthetic"]) == 0
    run = _read_run(output)
    assert run["pdf"]["searchable_pages"] == 2
    with pymupdf.open(output / "book.pdf") as document:
        assert document[0].search_for("校园阅读") and document[1].search_for("城市雨水")
        assert not document[0].search_for("城市雨水")
    assert main(["verify-pdf", "--input", str(output / "book.pdf"), "--expected-text", str(output / "book.txt"),
                 "--keyword", "城市雨水"]) == 0
