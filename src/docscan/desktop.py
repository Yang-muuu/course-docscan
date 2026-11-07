"""Single-window scanner for Windows."""

from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import filedialog, messagebox, ttk
from tkinter.font import nametofont

import numpy as np
from PIL import Image

from .common import DocscanError
from .config import load_config
from .desktop_preview import ImagePreview
from .desktop_scan import PhotoInput, PhotoProblem, Point, ScanCancelled, ScanPreview, prepare_photos, save_scans
from .geometry import validate_quad
from .image_io import IMAGE_SUFFIXES, load_photo

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/synthetic/sample.jpg"
MODES = {"普通纸张": "single", "摊开书籍": "spread", "同页多拍": "same"}
SIDES = {"single": "整页", "left": "左页", "right": "右页", "merged": "合成页"}
DEFAULT_CORNERS = ((0.06, 0.06), (0.94, 0.06), (0.94, 0.94), (0.06, 0.94))
LOG = logging.getLogger(__name__)
ERRORS = {
    "image_read_failed": "无法打开这张照片。请确认文件完整，或重新导入原始 JPG、PNG。",
    "unsupported_image": "请选用 JPG 或 PNG 照片。",
    "invalid_corners": "四角需要顺时针围成一个完整的纸面。请重新拖动圆点。",
    "empty_paper_mask": "未能找到纸面。请调整边界，或重新拍摄完整纸张。",
    "page_too_small": "照片尺寸太小，请导入相机原图。",
    "gutter_not_found": "未能确认书脊。请拖动分割线的两端，再点「开始处理」。",
    "invalid_gutter": "分割线过于靠边，请向书脊方向移动。",
    "incomplete_capture": "合成需要每张照片都拍到完整纸页。请调整四角或重新拍摄。",
    "capture_mismatch": "无法确认是同一页。请核对页码和文字，只保留同一页的 2 到 4 张照片。",
    "no_searchable_text": "没有识别到文字。请核对清晰度，或取消勾选可搜索文字后保存。",
    "pdf_font_missing_glyph": "有字符无法写入文字层。取消勾选可搜索文字后可保存图像 PDF。",
    "ocr_model_missing": "未找到 OCR 权重。请重新解压完整程序包。",
    "model_file_changed": "OCR 权重校验失败。请重新解压完整程序包。",
    "ocr_model_load_failed": "OCR 初始化失败。请重新解压程序，也可先保存图像 PDF。",
    "pdf_export_failed": "保存失败。请关闭同名 PDF，或换一个可写的文件夹后重试。",
}


def configuration() -> dict:
    config, _ = load_config()
    bundled = ROOT / "models"
    config["ocr"]["model_dir"] = str(bundled if bundled.is_dir() else ROOT / ".models")
    return config


def _configure_style(root: tk.Tk) -> None:
    root.configure(background="#f2f5fa")
    nametofont("TkDefaultFont", root=root).configure(family="Microsoft YaHei UI", size=10)
    nametofont("TkTextFont", root=root).configure(family="Microsoft YaHei UI", size=10)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#f2f5fa")
    style.configure("Panel.TFrame", background="#ffffff")
    style.configure("TLabel", background="#f2f5fa", foreground="#253a53", font=("Microsoft YaHei UI", 10))
    style.configure("Panel.TLabel", background="#ffffff", foreground="#253a53",
                    font=("Microsoft YaHei UI", 10, "bold"))
    style.configure("Title.TLabel", font=("Microsoft YaHei UI", 21, "bold"), foreground="#142b47")
    style.configure("Quiet.TLabel", foreground="#4f637c")
    style.configure("Warning.TLabel", foreground="#8f4108")
    style.configure("TButton", padding=(9, 6))
    style.configure("Primary.TButton", background="#1761be", foreground="white", bordercolor="#1761be",
                    lightcolor="#1761be", darkcolor="#1761be", padding=(14, 7))
    style.map("Primary.TButton", background=[("disabled", "#d5deea"), ("active", "#0d4c9b")],
              foreground=[("disabled", "#465972"), ("!disabled", "white")])
    style.configure("TCheckbutton", background="#f2f5fa", foreground="#253a53")
    style.configure("Horizontal.TProgressbar", background="#1761be", troughcolor="#e0e8f3")


class DocScanWindow:
    def __init__(self, root: tk.Tk, config: dict):
        self.root = root
        self.config = config
        self.photos: list[PhotoInput] = []
        self.hints: dict[tuple[Path, int], PhotoInput] = {}
        self.preview: ScanPreview | None = None
        self.workspace: TemporaryDirectory | None = None
        self.last_pdf: Path | None = None
        self.cancel_event: threading.Event | None = None
        self._closing = False
        self._editing = False
        self._messages = queue.SimpleQueue()
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.side = tk.StringVar()
        self.zoomed = tk.BooleanVar()
        self.searchable = tk.BooleanVar()
        self.status = tk.StringVar(value="添加照片，或打开示例开始。")
        self.issue = tk.StringVar()
        self.count = tk.StringVar(value="照片 · 0")
        self.percent = tk.DoubleVar()
        self._shown_photo: tuple[Path, int] | None = None
        self._shown_result: Path | None = None

        root.title("文档扫描 · DocScan")
        root.minsize(900, 600)
        width = min(1230, max(900, root.winfo_screenwidth() - 70))
        height = min(840, max(600, root.winfo_screenheight() - 110))
        root.geometry(f"{width}x{height}")
        _configure_style(root)
        self._layout()
        self._controls()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.report_callback_exception = self._callback_error
        root.bind("<Control-o>", lambda _event: self.choose_photos() if not self.busy else None)
        root.bind("<Control-s>", lambda _event: self.choose_pdf() if self.preview and not self.busy else None)
        root.bind("<Escape>", lambda _event: self.cancel())

    @property
    def busy(self) -> bool:
        return self.cancel_event is not None

    def _layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        mast = ttk.Frame(root, padding=(20, 13, 20, 6))
        mast.grid(row=0, column=0, sticky="ew")
        ttk.Label(mast, text="文档扫描", style="Title.TLabel").pack(side="left")
        ttk.Label(mast, text="300 DPI  ·  离线处理", style="Quiet.TLabel").pack(side="right")
        ttk.Label(mast, text="添加照片  /  查看效果  /  保存 PDF", style="Quiet.TLabel").pack(side="left", padx=20)

        toolbar = ttk.Frame(root, padding=(20, 7, 20, 9))
        toolbar.grid(row=1, column=0, sticky="ew")
        self.add_button = ttk.Button(toolbar, text="添加照片", command=self.choose_photos)
        self.example_button = ttk.Button(toolbar, text="打开示例", command=lambda: self.add_photos((EXAMPLE,)))
        self.add_button.pack(side="left", padx=(0, 5))
        self.example_button.pack(side="left", padx=(0, 15))
        ttk.Label(toolbar, text="扫描模式").pack(side="left", padx=(0, 5))
        self.mode_box = ttk.Combobox(toolbar, textvariable=self.mode, values=list(MODES), state="readonly", width=10)
        self.mode_box.pack(side="left", padx=(0, 7))
        self.mode_box.bind("<<ComboboxSelected>>", self._mode_changed)
        self.process_button = ttk.Button(toolbar, text="开始处理", command=self.process, style="Primary.TButton")
        self.process_button.pack(side="left")
        self.save_button = ttk.Button(toolbar, text="保存 PDF", command=self.choose_pdf, style="Primary.TButton")
        self.save_button.pack(side="right")
        self.ocr_check = ttk.Checkbutton(toolbar, text="生成可搜索文字", variable=self.searchable)
        self.ocr_check.pack(side="right", padx=(4, 10))

        area = ttk.Frame(root, padding=(20, 0, 20, 0))
        area.grid(row=2, column=0, sticky="nsew")
        area.rowconfigure(0, weight=1)
        area.columnconfigure(1, weight=1, uniform="preview")
        area.columnconfigure(2, weight=1, uniform="preview")
        sidebar = ttk.Frame(area, padding=9, style="Panel.TFrame")
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        ttk.Label(sidebar, textvariable=self.count, style="Panel.TLabel").pack(anchor="w", pady=(2, 8))
        listing = ttk.Frame(sidebar, style="Panel.TFrame")
        listing.pack(fill="both", expand=True)
        self.listing = tk.Listbox(listing, width=19, selectmode="browse", exportselection=False,
                                  borderwidth=0, highlightthickness=1, highlightbackground="#dfe7f0",
                                  background="white", foreground="#253a53", selectbackground="#d8e9ff",
                                  selectforeground="#18365c", activestyle="none")
        bar = ttk.Scrollbar(listing, orient="vertical", command=self.listing.yview)
        self.listing.configure(yscrollcommand=bar.set)
        self.listing.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.listing.bind("<<ListboxSelect>>", self._selection_changed)
        small = ttk.Frame(sidebar, style="Panel.TFrame", padding=(0, 8, 0, 0))
        small.pack(fill="x")
        self.up_button = ttk.Button(small, text="上移", width=6, command=lambda: self.move(-1))
        self.down_button = ttk.Button(small, text="下移", width=6, command=lambda: self.move(1))
        self.turn_button = ttk.Button(small, text="旋转 90°", width=6, command=self.rotate)
        self.remove_button = ttk.Button(small, text="删除", width=6, command=self.remove)
        for i, button in enumerate((self.up_button, self.down_button, self.turn_button, self.remove_button)):
            button.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
            small.columnconfigure(i % 2, weight=1)

        original, original_header = self._panel(area, 1, "原图")
        self.edit_button = ttk.Button(original_header, text="调整边界", command=self.edit)
        self.auto_button = ttk.Button(original_header, text="恢复自动", command=self.restore_auto)
        self.auto_button.pack(side="right", padx=(4, 0))
        self.edit_button.pack(side="right")
        self.original = ImagePreview(original, "添加照片后在这里预览")
        self.original.pack(fill="both", expand=True)

        result, result_header = self._panel(area, 2, "处理结果")
        self.zoom_check = ttk.Checkbutton(result_header, text="100% 查看", variable=self.zoomed, command=self._zoom)
        self.zoom_check.pack(side="right")
        self.side_box = ttk.Combobox(result_header, textvariable=self.side, state="readonly", width=6)
        self.side_box.pack(side="right", padx=6)
        self.side_box.bind("<<ComboboxSelected>>", lambda _event: self._show_result())
        self.result = ImagePreview(result, "点击「开始处理」生成页图预览")
        self.result.pack(fill="both", expand=True)

        footer = ttk.Frame(root, padding=(20, 8, 20, 11))
        footer.grid(row=3, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.issue, style="Warning.TLabel", wraplength=850).pack(anchor="w", pady=(0, 3))
        row = ttk.Frame(footer)
        row.pack(fill="x")
        ttk.Label(row, textvariable=self.status).pack(side="left")
        self.folder_button = ttk.Button(row, text="打开所在文件夹", command=lambda: self._open_output(folder=True))
        self.pdf_button = ttk.Button(row, text="打开 PDF", command=self._open_output)
        self.cancel_button = ttk.Button(row, text="取消", command=self.cancel)
        for button in (self.folder_button, self.pdf_button, self.cancel_button):
            button.pack(side="right", padx=(5, 0))
        ttk.Progressbar(footer, variable=self.percent, maximum=100).pack(fill="x", pady=(7, 0))

    @staticmethod
    def _panel(area, column: int, title: str):
        panel = ttk.Frame(area, padding=8, style="Panel.TFrame")
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 10 if column == 1 else 0))
        header = ttk.Frame(panel, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text=title, style="Panel.TLabel").pack(side="left", padx=(2, 5))
        return panel, header

    def _selected(self) -> int | None:
        selection = self.listing.curselection()
        return selection[0] if selection else None

    def _controls(self) -> None:
        index = self._selected()
        enabled = not self.busy
        for widget, available in (
            (self.add_button, True), (self.example_button, EXAMPLE.is_file()),
            (self.process_button, bool(self.photos)), (self.save_button, self.preview is not None),
            (self.ocr_check, True), (self.turn_button, index is not None),
            (self.remove_button, index is not None), (self.up_button, index is not None and index > 0),
            (self.down_button, index is not None and index + 1 < len(self.photos)),
            (self.edit_button, self.original.image is not None),
            (self.auto_button, index is not None and bool(self.photos[index].corners or self.photos[index].gutter)),
            (self.pdf_button, self.last_pdf is not None), (self.folder_button, self.last_pdf is not None),
        ):
            widget.configure(state="normal" if enabled and available else "disabled")
        self.mode_box.configure(state="readonly" if enabled else "disabled")
        self.listing.configure(state="normal" if enabled else "disabled")
        self.cancel_button.configure(state="normal" if self.cancel_event and not self.cancel_event.is_set() else "disabled")

    def choose_photos(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.root, title="添加照片",
                                           filetypes=[("照片", ("*.jpg", "*.jpeg", "*.png"))])
        self.add_photos(paths)

    def add_photos(self, paths) -> None:
        if self.busy:
            return
        entries = [PhotoInput(Path(path)) for path in paths if Path(path).suffix.lower() in IMAGE_SUFFIXES]
        if entries:
            first = len(self.photos)
            self.photos.extend(entries)
            self._clear_preview()
            self._list_photos(first)
            self.status.set(f"已添加 {len(self.photos)} 张照片，点击「开始处理」。")

    def _list_photos(self, index: int | None) -> None:
        self.listing.delete(0, "end")
        for i, photo in enumerate(self.photos, 1):
            self.listing.insert("end", f"{i:02d}  {photo.path.name}")
        self.count.set(f"照片 · {len(self.photos)}")
        if index is not None and self.photos:
            self.listing.selection_set(index)
            self.listing.see(index)
        self._selection_changed()

    def _selection_changed(self, _event=None) -> None:
        index = self._selected()
        item = self.photos[index] if index is not None else None
        key = (item.path, item.turns) if item else None
        self.issue.set("")
        self._editing = False
        self.original.edit()
        if key != self._shown_photo:
            self._shown_photo = key
            image = None
            if item:
                try:
                    pixels = np.rot90(load_photo(item.path).bgr, -item.turns)
                    image = Image.fromarray(pixels[:, :, ::-1])
                except DocscanError as error:
                    self.issue.set(ERRORS.get(error.code, str(error)))
            self.original.show(image)
        pages = self._result_pages()
        sides = [SIDES[page.side] for page in pages]
        self.side_box.configure(values=sides, state="readonly" if len(sides) > 1 else "disabled")
        self.side.set(sides[0] if sides else "")
        self._show_result()
        self._controls()

    def _result_pages(self):
        index = self._selected()
        if self.preview is None or index is None:
            return ()
        return tuple(page for page in self.preview.pages if self.preview.merged or page.photo_index == index)

    def _show_result(self) -> None:
        page = next((page for page in self._result_pages() if SIDES[page.side] == self.side.get()), None)
        path = page.source.image if page else None
        if path != self._shown_result:
            self._shown_result = path
            if path is None:
                self.result.show(None)
            else:
                with Image.open(path) as image:
                    self.result.show(image.convert("RGB"))
        if page:
            self.issue.set(page.warning)

    def _clear_preview(self) -> None:
        self.preview = None
        self._shown_result = None
        self.side.set("")
        self.side_box.configure(values=(), state="disabled")
        self.result.show(None)
        self.percent.set(0)
        if self.workspace is not None:
            self.workspace.cleanup()
            self.workspace = None

    def _mode_changed(self, _event=None) -> None:
        self.hints.clear()
        self._clear_preview()
        self._selection_changed()
        self.status.set(f"已选择「{self.mode.get()}」，点击「开始处理」。")

    def move(self, direction: int) -> None:
        index = self._selected()
        if not self.busy and index is not None and 0 <= index + direction < len(self.photos):
            target = index + direction
            self.photos[index], self.photos[target] = self.photos[target], self.photos[index]
            self._clear_preview()
            self._list_photos(target)
            self.status.set("页序已调整，点击「开始处理」更新预览。")

    def rotate(self) -> None:
        index = self._selected()
        if not self.busy and index is not None:
            self.photos[index] = self.photos[index].turned()
            self._clear_preview()
            self._selection_changed()
            self.status.set("已旋转，点击「开始处理」更新预览。")

    def remove(self) -> None:
        index = self._selected()
        if not self.busy and index is not None:
            del self.photos[index]
            self._clear_preview()
            self._list_photos(min(index, len(self.photos) - 1) if self.photos else None)
            self.status.set("照片已移除。" if self.photos else "添加照片，或打开示例开始。")

    def _zoom(self) -> None:
        self.original.zoom(self.zoomed.get())
        self.result.zoom(self.zoomed.get())

    def edit(self) -> None:
        index = self._selected()
        if index is None or self.original.image is None or self.busy:
            return
        item = self.photos[index]
        hint = self.hints.get((item.path, item.turns), item)
        spread = MODES[self.mode.get()] == "spread"
        if spread:
            top, bottom = item.gutter or hint.gutter or (0.5, 0.5)
            points = ((top, 0.035), (bottom, 0.965))
            self.issue.set("拖动分割线的两个圆点对准书脊。")
        else:
            points = item.corners or hint.corners or DEFAULT_CORNERS
            self.issue.set("按左上、右上、右下、左下拖动圆点，框住纸面。")
        self._editing = True
        self.original.edit(points, gutter=spread, changed=lambda handles: self._adjust(index, handles, spread))

    def _adjust(self, index: int, points: tuple[Point, ...], spread: bool) -> bool:
        if self.busy or not self._editing or index != self._selected():
            return False
        item = self.photos[index]
        try:
            if spread:
                item = replace(item, gutter=(points[0][0], points[1][0]))
            else:
                validate_quad(np.asarray(points) * 999, (1000, 1000))
                item = replace(item, corners=points)
        except DocscanError as error:
            self.issue.set(ERRORS[error.code])
            return False
        self.photos[index] = item
        self._clear_preview()
        self.issue.set("边界已调整，点击「开始处理」预览。")
        self._controls()
        return True

    def restore_auto(self) -> None:
        index = self._selected()
        if index is not None and not self.busy:
            self.photos[index] = replace(self.photos[index], corners=None, gutter=None)
            self._clear_preview()
            self._selection_changed()
            self.status.set("已恢复自动识别，点击「开始处理」。")

    def process(self) -> None:
        if not self.photos or self.busy:
            return
        mode, photos = MODES[self.mode.get()], tuple(self.photos)
        temporary = TemporaryDirectory(prefix="docscan-pages-")
        self._clear_preview()
        self.original.edit()

        def done(result: ScanPreview) -> None:
            self._clear_preview()
            self.preview, self.workspace = result, temporary
            self.hints = {(item.path, item.turns): item for item in result.hints}
            self._selection_changed()
            self.percent.set(100)
            self.status.set(f"处理完成 · {len(result.pages)} 页，可以保存 PDF。")

        self._start(lambda progress, cancel: prepare_photos(
            photos, mode, Path(temporary.name), self.config, progress, cancel,
        ), done, temporary.cleanup)

    def choose_pdf(self) -> None:
        if self.preview is None or self.busy:
            return
        target = filedialog.asksaveasfilename(parent=self.root, title="保存 PDF", defaultextension=".pdf",
                                              initialfile="扫描文档.pdf", filetypes=[("PDF 文档", "*.pdf")])
        if target:
            destination = Path(target)
            if destination.suffix.lower() != ".pdf":
                self.issue.set("请使用 .pdf 扩展名保存。")
                return
            preview, searchable = self.preview, self.searchable.get()

            def done(report) -> None:
                self.last_pdf = destination
                count = report["searchable_pages"]
                label = f" · {count} 页可搜索" if count else ""
                self.status.set(f"已保存 {report['pages']} 页 PDF{label}")
                self.percent.set(100)

            self._start(lambda progress, cancel: save_scans(
                preview, destination, searchable, self.config, progress, cancel,
            ), done)

    def _start(self, task, done, discard=None) -> None:
        cancel = self.cancel_event = threading.Event()
        self.issue.set("")
        self.status.set("正在准备…")
        self.percent.set(0)
        self._controls()

        def worker() -> None:
            try:
                data = task(lambda phase, i, n: self._messages.put(("progress", (phase, i, n))), cancel)
                self._messages.put(("done", data))
            except ScanCancelled:
                self._messages.put(("cancelled", None))
            except Exception as error:
                LOG.exception("Desktop operation failed")
                self._messages.put(("error", error))

        threading.Thread(target=worker, name="DocScan worker").start()
        self.root.after(60, self._poll, done, discard)

    def _poll(self, done, discard) -> None:
        while not self._messages.empty():
            kind, data = self._messages.get()
            if kind == "progress":
                if self.cancel_event and not self.cancel_event.is_set():
                    phase, index, count = data
                    self.status.set(f"{phase} · {index} / {count}")
                    self.percent.set(100 * (index - 1) / count)
                continue
            self.cancel_event = None
            if self._closing:
                if discard:
                    discard()
                self.close()
                return
            self._controls()
            if kind == "done":
                done(data)
            else:
                if discard:
                    discard()
                self.percent.set(0)
                if kind == "cancelled":
                    self.status.set("已取消。")
                else:
                    self.status.set("处理未完成")
                    self._show_problem(data)
            self._controls()
            return
        self.root.after(60, self._poll, done, discard)

    def _show_problem(self, error: Exception) -> None:
        if isinstance(error, PhotoProblem):
            index = error.photo_index
            self.listing.selection_clear(0, "end")
            self.listing.selection_set(index)
            self.listing.see(index)
            self._selection_changed()
        if isinstance(error, DocscanError):
            if error.code == "gutter_not_found":
                self.edit()
            prefix = f"第 {error.photo_index + 1} 张 · " if isinstance(error, PhotoProblem) else ""
            self.issue.set(prefix + ERRORS.get(error.code, str(error)))
        elif isinstance(error, OSError):
            self.issue.set(ERRORS["pdf_export_failed"])
        else:
            self.issue.set("处理失败。请重新选择清晰的照片后重试。")

    def cancel(self) -> None:
        if self.cancel_event:
            self.cancel_event.set()
            self.status.set("正在取消，等待当前页结束…")
            self._controls()

    def _open_output(self, folder: bool = False) -> None:
        if self.last_pdf is not None:
            try:
                os.startfile(self.last_pdf.parent if folder else self.last_pdf)
            except OSError:
                self.issue.set("无法打开保存位置，请检查文件是否已移动。")

    def _callback_error(self, kind, error, traceback) -> None:
        LOG.error("Window action failed", exc_info=(kind, error, traceback))
        self._show_problem(error)

    def close(self) -> None:
        if self.busy:
            self._closing = True
            self.cancel()
            return
        self._clear_preview()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        local = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DocScan"
        local.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(local / "paddlex"))
        logging.basicConfig(filename=local / "diagnostics.log", encoding="utf-8", level=logging.WARNING)
        DocScanWindow(root, configuration())
    except Exception:
        LOG.exception("DocScan could not start")
        messagebox.showerror("文档扫描 · 启动失败", "请完整解压程序包后重新启动。", parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
