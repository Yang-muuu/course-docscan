import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from docscan.cli import main
from docscan.common import dependency_versions
from docscan.config import DEFAULTS
from docscan.export_pdf import export_pdf, verify_pdf
from docscan.image_io import NORMALIZED_SPACE, load_photo, write_png
from docscan.models import model_status
from docscan.ocr import PaddleEngine, order_lines


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_batch_continues_and_separates_image_pdf(samples, tmp_path):
    inputs = tmp_path / "照片"
    inputs.mkdir()
    shutil.copyfile(samples / "sample.jpg", inputs / "01_照片.jpg")
    (inputs / "02_broken.jpg").write_bytes(b"broken jpeg")
    shutil.copyfile(samples / "syn-04-front_even.jpg", inputs / "03_照片.jpg")
    output = tmp_path / "run"
    assert main(["scan", "--input", str(inputs), "--method", "E", "--ocr", "none", "--pdf-mode", "image",
                 "--output", str(output), "--source", "synthetic"]) == 1
    run = read_json(output / "run.json")
    assert run["status"] == "incomplete" and run["passed"] == 2 and run["failed"] == 1
    assert [item["status"] for item in run["images"]] == ["ok", "error", "ok"]
    child = output / run["images"][0]["run_dir"]
    assert read_json(child / "run.json")["pdf"]["searchable"] is False
    assert not (child / "ocr.json").exists()
    with pymupdf.open(child / "scan.pdf") as pdf:
        assert not pdf[0].get_text().strip()
    assert main(["scan", "--input", str(inputs), "--ocr", "none", "--pdf-mode", "searchable",
                 "--output", str(tmp_path / "invalid")]) == 1
    assert not (tmp_path / "invalid").exists()


def test_reading_order_sorts_rows_and_fragments():
    lines = [
        {"text": "右", "polygon": [[90, 9], [120, 9], [120, 28], [90, 28]]},
        {"text": "第二行", "polygon": [[14, 41], [120, 41], [120, 60], [14, 60]]},
        {"text": "左", "polygon": [[14, 12], [46, 12], [46, 30], [14, 30]]},
    ]
    ordered, full_text = order_lines(lines)
    assert full_text == "左 右\n第二行"
    assert [line["reading_order"] for line in ordered] == [1, 2, 3]


def test_blank_photo_failure_and_explicit_manual_fallback(tmp_path):
    image, corners = tmp_path / "空白.png", tmp_path / "四角.json"
    write_png(image, np.full((240, 180, 3), 200, np.uint8))
    corners.write_text(json.dumps({"image_size": [180, 240], "coordinate_space": NORMALIZED_SPACE,
                                   "order": "tl_tr_br_bl", "points": [[10, 10], [170, 10], [170, 230], [10, 230]]}),
                       encoding="utf-8")
    args = ["scan", "--input", str(image), "--source", "synthetic", "--method", "E"]
    failed, recovered = tmp_path / "failed", tmp_path / "recovered"
    assert main([*args, "--output", str(failed)]) == 1
    assert read_json(failed / "run.json")["error_code"] == "detection_failed"
    assert not (failed / "warped_gray.png").exists()
    assert main([*args, "--corners", str(corners), "--auto-first", "--output", str(recovered)]) == 0
    run = read_json(recovered / "run.json")
    assert run["manual_fallback_used"] and run["geometry"]["auto_status"] == "detection_failed"
    assert run["geometry"]["corner_origin"] == "manual_fallback"


def test_missing_model_is_error_even_on_blank(tmp_path):
    engine = PaddleEngine({**DEFAULTS["ocr"], "model_dir": str(tmp_path / "absent")})
    result = engine.recognize(np.full((240, 360), 255, np.uint8), NORMALIZED_SPACE)
    assert result["status"] == "error" and result["error_code"] in {"ocr_dependency_missing", "ocr_model_missing"}
    assert result["inference_attempted"] is False


def test_pdf_geometry_with_explicit_test_text(tmp_path):
    # Hand-specified polygons test the PDF adapter without claiming OCR inference.
    image = np.full((120, 250), 230, dtype=np.uint8)
    annotation = {"status": "ok", "image_size": [250, 120], "full_text": "中文文档\nSearch 42",
                  "lines": [{"text": "中文文档", "polygon": [[14, 15], [120, 15], [120, 38], [14, 38]]},
                            {"text": "Search 42", "polygon": [[18, 58], [125, 69], [122, 90], [16, 79]]}]}
    target, background = tmp_path / "sample.pdf", tmp_path / "scan.png"
    write_png(background, image)
    assert export_pdf(target, image, annotation, DEFAULTS["pdf"])["searchable"]
    report = verify_pdf(target, annotation["full_text"], "中文", annotation, background)
    assert report["keyword_hits"] == 1 and report["position_checked"] and report["render_equals_image_only"]


@pytest.mark.ocr_integration
def test_paddle_real_inference_and_blank(samples, tmp_path):
    versions = dependency_versions()
    if any(not versions[p] for p in ("paddlepaddle", "paddleocr", "paddlex")):
        pytest.skip("Paddle inference dependencies are absent")
    if not all(model["ready"] for model in model_status(Path(DEFAULTS["ocr"]["model_dir"])).values()):
        pytest.skip("Local Paddle models are absent; run prepare-models")
    engine = PaddleEngine({**DEFAULTS["ocr"], "max_input_side": 1100})
    photo = load_photo(samples / "syn-03-front_even.jpg")
    result = engine.recognize(cv2.cvtColor(photo.bgr, cv2.COLOR_BGR2GRAY), NORMALIZED_SPACE)
    assert result["status"] == "ok", result.get("error")
    assert "博物馆" in result["full_text"] and "Museum" in result["full_text"]
    assert result["model_input_size"] != result["image_size"]
    assert max(point[1] for line in result["lines"] for point in line["polygon"]) > result["model_input_size"][1]
    assert result["raw_result"]["res"]["rec_texts"]
    target = tmp_path / "real-model-synthetic-page.pdf"
    assert export_pdf(target, photo.bgr, result, DEFAULTS["pdf"])["searchable"]
    assert verify_pdf(target, result["full_text"], "博物馆", result, samples / "syn-03-front_even.jpg")["position_checked"]
    blank = engine.recognize(np.full((440, 650), 255, np.uint8), NORMALIZED_SPACE)
    assert blank["status"] == "no_text" and blank["full_text"] == "" and blank["inference_attempted"]
    assert blank["model_load_ms"] == result["model_load_ms"]
