import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymupdf
import cv2
from PIL import Image

from docscan.candidate_inputs import candidate_inputs
from docscan.candidates import _ink_strength, scan_candidates
from docscan.config import load_config


def test_candidate_archive_skips_each_reference_and_keeps_source_pixels(tmp_path):
    photo = BytesIO()
    Image.new("RGB", (96, 72), "lightgray").save(photo, format="JPEG")
    raw = photo.getvalue()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((100, 100), "Reference page")
        page = pdf.new_page()
        page.insert_image(pymupdf.Rect(20, 30, 116, 102), stream=raw)
        document = pdf.tobytes()
    path = tmp_path / "扫描样例.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("PDF 10.pdf", document)
        archive.writestr("PDF 2.pdf", document)

    items = list(candidate_inputs(path, 1800))
    assert [item.folder.as_posix() for item in items] == ["pdf-01/page-02", "pdf-02/page-02"]
    assert [item.source["name"] for item in items] == ["PDF 2.pdf", "PDF 10.pdf"]
    assert all(item.source["extraction"] == "embedded_image" and item.pixels.shape == (72, 96, 3) for item in items)
    assert all(item.source["sha256"] == hashlib.sha256(raw).hexdigest() for item in items)


def test_candidate_batch_keeps_three_outputs_and_reports_bad_photo(tmp_path, monkeypatch):
    class ExampleScanner:
        model_hashes = {}

        def __init__(self, config):
            pass

        def scan(self, image):
            return {label: np.full(image.shape[:2], 100 * i, np.uint8) for i, label in enumerate("ABC")}, {}

    monkeypatch.setattr("docscan.candidates.CandidateScanner", ExampleScanner)
    source = tmp_path / "输入"
    source.mkdir()
    (source / "01.jpg").write_bytes(b"invalid jpg")
    Image.new("RGB", (64, 64), "white").save(source / "02.png")
    config, _ = load_config()
    result = scan_candidates(source, tmp_path / "输出", config)
    root = Path(result["run_dir"])
    assert (result["status"], result["photos"], result["processed"], result["images"]) == ("error", 2, 1, 3)
    assert sorted(path.name for path in (root / "image-0002").iterdir()) == ["A.png", "B.png", "C.png"]
    assert not (root / "image-0001").exists()
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert run["photos"][0]["source"]["error_code"] == "image_read_failed"


def test_warm_shadow_reaching_page_edge_keeps_formula_ink():
    image = np.full((640, 440, 3), 234, np.uint8)
    image[:150] = (60, 82, 104)
    cv2.putText(image, "x + y = 12", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 38, 47), 1)
    ink, strong, _ = _ink_strength(image, None)
    assert np.count_nonzero(ink[45:80] > 0.16) >= 55
    assert np.count_nonzero(strong[45:80] > 0.16) >= 55
    assert np.median(ink[:100]) == 0
