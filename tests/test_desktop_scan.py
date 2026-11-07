from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from docscan.config import load_config
from docscan.desktop_scan import PhotoInput, PagePreview, ScanCancelled, ScanPreview, _may_be_blurred, save_scans
from docscan.export_pdf import PdfSource
from docscan.image_io import write_png


def test_turning_a_manually_cropped_page_keeps_corner_order():
    corners = ((0.12, 0.13), (0.88, 0.16), (0.84, 0.89), (0.10, 0.91))
    original = PhotoInput(Path("文档.jpg"), corners=corners, gutter=(0.45, 0.49))
    rotated = original.turned()
    assert rotated.turns == 1
    assert np.allclose(rotated.corners, ((0.09, 0.10), (0.87, 0.12), (0.84, 0.88), (0.11, 0.84)))
    assert rotated.gutter is None


def test_cancel_during_pdf_build_leaves_destination_untouched(tmp_path):
    raster = tmp_path / "page.png"
    write_png(raster, np.full((240, 180, 3), 240, np.uint8))
    preview = ScanPreview((PagePreview(PdfSource(raster, None, (43.2, 57.6)), 0, "single"),), ())
    destination = tmp_path / "已有.pdf"
    destination.write_bytes(b"previous file")
    cancel = Event()
    config, _ = load_config()

    with pytest.raises(ScanCancelled):
        save_scans(preview, destination, False, config, lambda *_: cancel.set(), cancel)

    assert destination.read_bytes() == b"previous file"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["page.png", "已有.pdf"]


def test_blur_warning_distinguishes_clear_print_from_defocus():
    page = np.full((820, 600, 3), 255, np.uint8)
    for y in range(80, 750, 40):
        cv2.putText(page, "PRINTED TEXT 1208", (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    assert not _may_be_blurred(page)
    assert _may_be_blurred(cv2.GaussianBlur(page, (0, 0), 4))
