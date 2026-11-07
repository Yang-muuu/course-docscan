"""Image canvas with zoom and draggable scan boundaries."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from PIL import Image, ImageTk

from .desktop_scan import Point


class ImagePreview(ttk.Frame):
    def __init__(self, parent, empty: str):
        super().__init__(parent, style="Panel.TFrame")
        self.empty = empty
        self.image: Image.Image | None = None
        self.bitmap: ImageTk.PhotoImage | None = None
        self.actual_size = False
        self.origin = (0, 0)
        self.extent = (1, 1)
        self.handles: list[Point] = []
        self.gutter = False
        self._change: Callable[[tuple[Point, ...]], bool] | None = None
        self._drag: int | None = None
        self._previous: tuple[Point, ...] = ()
        self._redraw_id = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, background="#e8edf4", highlightthickness=0, width=100, height=140)
        xbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xbar.grid(row=1, column=0, sticky="ew")
        ybar.grid(row=0, column=1, sticky="ns")
        self.canvas.bind("<Configure>", self._schedule_redraw)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Shift-MouseWheel>", lambda event: self._wheel(event, horizontal=True))

    def show(self, image: Image.Image | None) -> None:
        self.image = image
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.edit()
        self._schedule_redraw()

    def zoom(self, actual_size: bool) -> None:
        self.actual_size = actual_size
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._schedule_redraw()

    def edit(self, points: tuple[Point, ...] = (), *, gutter: bool = False,
             changed: Callable[[tuple[Point, ...]], bool] | None = None) -> None:
        self.handles = list(points)
        self.gutter = gutter
        self._change = changed
        self._drag = None
        self.canvas.configure(cursor="crosshair" if points else "")
        self._draw_handles()

    def _schedule_redraw(self, _event=None) -> None:
        if self._redraw_id is not None:
            self.after_cancel(self._redraw_id)
        self._redraw_id = self.after(35, self._redraw)

    def _redraw(self) -> None:
        self._redraw_id = None
        canvas = self.canvas
        canvas.delete("all")
        self.bitmap = None
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if self.image is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(width / 2, height / 2, text=self.empty, fill="#52657b",
                               font=("Microsoft YaHei UI", 11), width=max(80, width - 36))
            return
        w, h = self.image.size
        scale = 1 if self.actual_size else min(1, max(1, width - 20) / w, max(1, height - 20) / h)
        display = (max(1, round(w * scale)), max(1, round(h * scale)))
        raster = self.image if display == (w, h) else self.image.resize(display, Image.Resampling.LANCZOS)
        self.bitmap = ImageTk.PhotoImage(raster, master=canvas)
        self.origin = (max(0, (width - display[0]) // 2), max(0, (height - display[1]) // 2))
        self.extent = (max(1, display[0] - 1), max(1, display[1] - 1))
        canvas.configure(scrollregion=(0, 0, max(width, display[0]), max(height, display[1])))
        canvas.create_image(*self.origin, image=self.bitmap, anchor="nw")
        self._draw_handles()

    def _point(self, point: Point) -> Point:
        return self.origin[0] + point[0] * self.extent[0], self.origin[1] + point[1] * self.extent[1]

    def _draw_handles(self) -> None:
        self.canvas.delete("edge")
        if not self.handles or self.bitmap is None:
            return
        points = [self._point(point) for point in self.handles]
        if self.gutter:
            ends = self._point((self.handles[0][0], 0)), self._point((self.handles[1][0], 1))
            self.canvas.create_line(*ends[0], *ends[1], fill="#1367d4", width=3, tags="edge")
        else:
            self.canvas.create_polygon(points, outline="#1367d4", fill="", width=3, tags="edge")
        for x, y in points:
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#ffffff", outline="#1367d4",
                                    width=3, tags="edge")

    def _press(self, event) -> None:
        if not self.handles:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        for index, handle in enumerate(self.handles):
            hx, hy = self._point(handle)
            if (hx - x) ** 2 + (hy - y) ** 2 <= 17 ** 2:
                self._drag = index
                self._previous = tuple(self.handles)
                break

    def _move(self, event) -> None:
        if self._drag is None:
            return
        x = (self.canvas.canvasx(event.x) - self.origin[0]) / self.extent[0]
        y = (self.canvas.canvasy(event.y) - self.origin[1]) / self.extent[1]
        if self.gutter:
            point = (min(0.95, max(0.05, x)), self.handles[self._drag][1])
        else:
            point = (min(1, max(0, x)), min(1, max(0, y)))
        self.handles[self._drag] = point
        self._draw_handles()

    def _release(self, _event) -> None:
        if self._drag is None:
            return
        self._drag = None
        points = tuple(self.handles)
        if points != self._previous and self._change is not None and not self._change(points):
            self.handles = list(self._previous)
        self._draw_handles()

    def _wheel(self, event, *, horizontal: bool = False) -> str:
        move = self.canvas.xview_scroll if horizontal else self.canvas.yview_scroll
        move(-int(event.delta / 120), "units")
        return "break"

    def destroy(self) -> None:
        if self._redraw_id is not None:
            self.after_cancel(self._redraw_id)
        super().destroy()
