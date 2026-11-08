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

from docscan.desktop import EXAMPLES, ROOT, VARIANTS, DocScanWindow, configuration


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


def verify_shadow_choices(app: DocScanWindow, work: Path, screenshot: Path | None) -> dict:
    assert not hasattr(app, "mode_box")
    app.example_button.invoke()
    assert tuple(item.path for item in app.photos) == EXAMPLES and len(EXAMPLES) == 3
    assert len(app.original.handles) == 4, "A boundary must be shown immediately after import"
    wait_ui(app)
    for index, item in enumerate(app.photos):
        assert item.boundary is not None and item.corners is None
        select(app, index)
        assert tuple(app.original.handles) == item.boundary.points
    assert app.photos[2].boundary.analysis.geometry.mode == "book_leaf"

    before = tuple(app.original.handles)
    drag_handle(app)
    assert app.photos[2].corners and tuple(app.original.handles) != before
    app.auto_button.invoke()
    assert app.photos[2].corners is None and tuple(app.original.handles) == before
    drag_handle(app)
    assert app.photos[2].corners == tuple(app.original.handles)

    folder = work / "中文 输入"
    folder.mkdir()
    extra = folder / "复查边界.jpg"
    shutil.copy2(EXAMPLES[0], extra)
    add(app, extra)
    assert len(app.original.handles) == 4
    wait_ui(app)
    assert app.photos[3].boundary is not None
    app.turn_button.invoke()
    wait_ui(app)
    assert app.photos[3].turns == 1 and len(app.original.handles) == 4
    assert app.photos[3].boundary is not None
    app.remove_button.invoke()

    process(app, 3)
    assert all(len(page.variants) == 3 and len({path.read_bytes() for path in page.variants}) == 3
               for page in app.preview.pages)
    assert app.photos[2].corners is not None
    chosen = []
    for index, page in enumerate(app.preview.pages):
        select(app, index)
        choice = (index + 1) % 3
        app.variant_box.current(choice)
        app.variant_box.event_generate("<<ComboboxSelected>>")
        app.root.update()
        chosen.append(page.variants[choice])
    select(app, 0)
    assert app.variant.get() == VARIANTS[1] and app._shown_result == chosen[0]
    app.zoom_check.invoke()
    wait_ui(app)
    assert app.result.extent[0] > app.result.canvas.winfo_width()
    app.zoom_check.invoke()

    result = work / "实拍去影选图.pdf"
    save(app, result)
    with pymupdf.open(result) as document:
        assert len(document) == len(chosen)
        for page, expected in zip(document, chosen, strict=True):
            raster = pymupdf.Pixmap(document, page.get_images()[0][0])
            mode = "L" if raster.n == 1 else "RGB"
            with Image.open(expected) as image:
                assert raster.samples == image.convert(mode).tobytes()
            assert abs(raster.width * 72 / page.rect.width - 300) < 1

    app.ocr_check.invoke()
    cancelled = work / "取消保存.pdf"
    sentinel = result.read_bytes()
    cancelled.write_bytes(sentinel)
    save(app, cancelled, cancel=True)
    assert app.status.get() == "已取消。" and cancelled.read_bytes() == sentinel
    searchable = work / "实拍可搜索.pdf"
    save(app, searchable)
    with pymupdf.open(searchable) as document:
        hits = [len(page.search_for(word)) for page, word in zip(document, ("颜色模型", "Phong", "区域"), strict=True)]
        assert all(hits), [page.get_text()[:130] for page in document]
    with patch("docscan.desktop.os.startfile") as open_file:
        app.pdf_button.invoke()
        app.folder_button.invoke()
        assert [call.args[0] for call in open_file.call_args_list] == [searchable, work]
    if screenshot:
        select(app, 2)
        app.root.attributes("-topmost", True)
        app.root.lift()
        app.root.update()
        ImageGrab.grab(window=int(app.root.wm_frame(), 0)).save(screenshot)
        app.root.attributes("-topmost", False)
    return {"single_workflow": True, "automatic_four_corners": True, "dragged_book_corners": True,
            "restored_automatic_boundary": True, "real_examples": [path.name for path in EXAMPLES],
            "selected_variants": [path.stem for path in chosen], "pdf_uses_selected_images": True,
            "dpi": 300, "chinese_search_hits": hits, "cancel_preserved_original": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    work = args.output.resolve()
    screenshot = args.screenshot.resolve() if args.screenshot else None
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
    os.chdir(ROOT)
    root = tk.Tk()
    app = DocScanWindow(root, configuration())
    try:
        choices = verify_shadow_choices(app, work, screenshot)
    finally:
        app.close()
        if app.busy:
            wait_ui(app)
    for name in ("cv2", "paddle", "paddleocr", "paddlex", "pymupdf", "tkinter"):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(runtime), name
    evidence = {"launcher": launch, "runtime": sys.executable, "isolated_imports": True, **choices}
    (work / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
