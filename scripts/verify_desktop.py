"""Exercise the built Windows program and save evidence outside the bundle."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from unittest.mock import patch

import psutil
import pymupdf
from PIL import Image, ImageGrab

from docscan.desktop import EXAMPLE, ROOT, DocScanWindow, configuration


def verify_launcher(work: Path) -> dict:
    user32 = ctypes.WinDLL("user32")
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = (callback, wintypes.LPARAM)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    launcher = subprocess.Popen([str(ROOT / "DocScan.exe")], cwd=work)
    try:
        deadline = time.monotonic() + 25
        window = None
        while window is None and time.monotonic() < deadline and launcher.poll() is None:
            children = {child.pid for child in psutil.Process(launcher.pid).children()}

            @callback
            def match(hwnd, _data):
                nonlocal window
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in children and user32.IsWindowVisible(hwnd):
                    title = ctypes.create_unicode_buffer(200)
                    user32.GetWindowTextW(hwnd, title, len(title))
                    if title.value == "文档扫描 · DocScan":
                        window = hwnd
                return True

            user32.EnumWindows(match, 0)
            if window is None:
                time.sleep(0.1)
        assert window is not None, "DocScan.exe did not open its desktop window"
        user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE
        assert launcher.wait(timeout=15) == 0
        return {"exe": str(ROOT / "DocScan.exe"), "visible_window": True}
    finally:
        if launcher.poll() is None:
            for child in psutil.Process(launcher.pid).children():
                child.terminate()
            launcher.terminate()
            launcher.wait(timeout=10)


def wait_ui(app: DocScanWindow, timeout: float = 120) -> None:
    limit = time.monotonic() + timeout
    while app.busy and time.monotonic() < limit:
        app.root.update()
        time.sleep(0.03)
    assert not app.busy, "Desktop operation timed out"
    # Allow the preview's throttled redraw to run.
    app.root.update()
    time.sleep(0.07)
    app.root.update()


def select(app: DocScanWindow, index: int) -> None:
    app.listing.selection_clear(0, "end")
    app.listing.selection_set(index)
    app.listing.event_generate("<<ListboxSelect>>")
    app.root.update()


def add(app: DocScanWindow, *paths: Path) -> None:
    with patch("docscan.desktop.filedialog.askopenfilenames", return_value=tuple(map(str, paths))):
        app.add_button.invoke()
    app.root.update()


def clear(app: DocScanWindow) -> None:
    while app.photos:
        app.remove_button.invoke()


def drag_handle(app: DocScanWindow) -> None:
    app.edit_button.invoke()
    wait_ui(app)
    canvas = app.original.canvas
    assert app.original.handles
    x, y = app.original._point(app.original.handles[0])
    canvas.event_generate("<ButtonPress-1>", x=round(x), y=round(y))
    canvas.event_generate("<B1-Motion>", x=round(x + 4), y=round(y + 3))
    canvas.event_generate("<ButtonRelease-1>", x=round(x + 4), y=round(y + 3))
    app.root.update()


def process(app: DocScanWindow, pages: int) -> None:
    app.process_button.invoke()
    wait_ui(app)
    assert app.preview is not None, app.issue.get()
    assert len(app.preview.pages) == pages, app.status.get()


def save(app: DocScanWindow, destination: Path, *, cancel: bool = False) -> None:
    with patch("docscan.desktop.filedialog.asksaveasfilename", return_value=str(destination)):
        app.save_button.invoke()
    if cancel:
        app.cancel_button.invoke()
    wait_ui(app)
    if not cancel:
        assert app.last_pdf == destination and destination.is_file(), app.issue.get()


def verify_paper(app: DocScanWindow, work: Path, screenshot: Path | None) -> dict:
    folder = work / "中文 输入"
    folder.mkdir()
    reading, rainfall = folder / "校园阅读.jpg", folder / "城市雨水.png"
    shutil.copy2(EXAMPLE, reading)
    with Image.open(ROOT / "examples/desktop/book-spread.jpg") as spread:
        spread.crop((spread.width - 840, 0, spread.width, spread.height)).save(rainfall)
    add(app, reading, rainfall, reading)
    select(app, 2)
    app.remove_button.invoke()
    assert len(app.photos) == 2
    select(app, 0)
    app.down_button.invoke()
    assert [item.path for item in app.photos] == [rainfall, reading]
    app.turn_button.invoke()
    assert app.photos[1].turns == 1
    for _ in range(3):
        app.turn_button.invoke()
    assert app.photos[1].turns == 0
    process(app, 2)
    drag_handle(app)
    assert app.photos[1].corners and app.preview is None, app.issue.get()
    process(app, 2)
    app.zoom_check.invoke()
    wait_ui(app)
    assert app.result.extent[0] > app.result.canvas.winfo_width()
    app.zoom_check.invoke()
    image = work / "批量扫描.pdf"
    save(app, image)
    with pymupdf.open(image) as document:
        assert len(document) == 2
        for page, expected in zip(document, app.preview.pages, strict=True):
            xref = page.get_images()[0][0]
            pixmap = pymupdf.Pixmap(document, xref)
            with Image.open(expected.source.image) as raster:
                assert pixmap.samples == raster.convert("RGB").tobytes()
            assert abs(pixmap.width * 72 / page.rect.width - 300) < 1

    app.ocr_check.invoke()
    cancelled = work / "取消时保留原文件.pdf"
    sentinel = image.read_bytes()
    cancelled.write_bytes(sentinel)
    save(app, cancelled, cancel=True)
    assert app.status.get() == "已取消。" and cancelled.read_bytes() == sentinel
    assert not list(work.glob(".docscan-*"))

    searchable = work / "可搜索扫描.pdf"
    save(app, searchable)
    with pymupdf.open(searchable) as document:
        assert len(document) == 2
        hits = [len(page.search_for(term)) for page, term in zip(document, ("城市雨水", "校园阅读"), strict=True)]
        assert all(hits), [page.get_text() for page in document]
    with patch("docscan.desktop.os.startfile") as open_file:
        app.pdf_button.invoke()
        app.folder_button.invoke()
        assert [call.args[0] for call in open_file.call_args_list] == [searchable, work]
    if screenshot is not None:
        wait_ui(app)
        app.root.attributes("-topmost", True)
        app.root.lift()
        app.root.update()
        hwnd = int(app.root.wm_frame(), 0)
        ImageGrab.grab(window=hwnd).save(screenshot)
        app.root.attributes("-topmost", False)
    app.ocr_check.invoke()
    return {"page_order": [rainfall.name, reading.name], "chinese_search_hits": hits,
            "adjusted_four_corners": True, "dpi": 300, "cancel_preserved_original": True}


def verify_other_modes(app: DocScanWindow, work: Path) -> dict:
    examples = ROOT / "examples/desktop"
    clear(app)
    app.mode_box.current(1)
    app.mode_box.event_generate("<<ComboboxSelected>>")
    add(app, examples / "book-spread.jpg")
    process(app, 2)
    assert [page.side for page in app.preview.pages] == ["left", "right"]
    drag_handle(app)
    assert app.photos[0].gutter and app.preview is None, app.issue.get()
    process(app, 2)
    book = work / "书页拆分.pdf"
    save(app, book)
    with pymupdf.open(book) as document:
        assert len(document) == 2

    clear(app)
    app.mode_box.current(2)
    app.mode_box.event_generate("<<ComboboxSelected>>")
    add(app, examples / "same-page-1.jpg", examples / "same-page-2.jpg")
    process(app, 1)
    merged = work / "同页合成.pdf"
    save(app, merged)
    with pymupdf.open(merged) as document:
        assert len(document) == 1

    clear(app)
    with Image.open(examples / "book-spread.jpg") as spread:
        other = Image.new("RGB", (984, 1332), (154, 116, 63))
        other.paste(spread.crop((spread.width - 840, 0, spread.width, spread.height)), (72, 72))
        mismatch = work / "不同内容.png"
        other.save(mismatch)
    add(app, examples / "same-page-1.jpg", mismatch)
    app.process_button.invoke()
    wait_ui(app)
    assert app.preview is None and "无法确认是同一页" in app.issue.get(), app.issue.get()
    return {"split_pages": 2, "adjusted_binding": True, "same_page_pdf": 1, "rejected_different_content": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    work = args.output.resolve()
    work.mkdir(parents=True)
    runtime = ROOT / "runtime"
    assert Path(sys.executable).resolve().is_relative_to(runtime), sys.executable
    assert all(Path(path).resolve().is_relative_to(ROOT) for path in sys.path if path), sys.path
    os.environ["PATH"] = os.pathsep.join((str(Path(os.environ["WINDIR"]) / "System32"), os.environ["WINDIR"]))
    assert shutil.which("python") is None
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(work / "paddlex")
    os.environ["LOCALAPPDATA"] = str(work / "appdata")
    os.environ["TEMP"] = os.environ["TMP"] = str(work)
    launch = verify_launcher(work)
    root = tk.Tk()
    app = DocScanWindow(root, configuration())
    try:
        paper = verify_paper(app, work, args.screenshot)
        others = verify_other_modes(app, work)
    finally:
        app.close()
        if app.busy:
            wait_ui(app)
    for name in ("cv2", "paddle", "paddleocr", "paddlex", "pymupdf", "tkinter"):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(runtime), name
    evidence = {"launcher": launch, "runtime": sys.executable, "isolated_imports": True, **paper, **others}
    (work / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
