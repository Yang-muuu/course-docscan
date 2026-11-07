from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import DocscanError, file_sha256, write_csv, write_json
from .dataset import CONDITIONS, MANIFEST_FIELDS
from .fonts import chinese_font, sample_font
from .geometry import CORNER_ORDER
from .image_io import NORMALIZED_SPACE

PAGE_SIZE = (840, 1188)
PHOTO_SIZE = (1100, 1430)
DOCUMENTS = (
    ("校园阅读计划", "Campus Reading Notes", (
        "每周三下午开放小组讨论。", "请带上阅读卡和铅笔。", "记录一个问题，再写下你的解释。",
        "The library opens at 9:00.", "Return borrowed books by Friday.", "下次活动将在东侧阅览室举行。",
        "借阅编号为 R-12，日期为十一月。", "Bring a short summary to the meeting.",
        "建议先浏览目录，再读第一章。", "备注：薄纸背面的字迹较浅。",
    )),
    ("城市雨水观察", "Rainfall Field Notes", (
        "同一地点连续记录四天的降雨。", "把量杯放在远离树叶的地方。", "十点读取刻度，记到毫米。",
        "Daily reading: 16 mm at noon.", "Label each bottle before storage.", "地面潮湿时，请沿步道行走。",
        "东侧花坛的积水消退较慢。", "Cloud cover changed after lunch.",
        "记录日期、地点与观察者。", "备注：浅灰线表示参考高度。",
    )),
    ("博物馆参观安排", "Museum Visit Schedule", (
        "团队在正门集合后统一入场。", "第一站参观陶器与青铜展厅。", "展品标签列出年代和出土地点。",
        "Meet at the main entrance at 10.", "Keep bags in the lockers upstairs.", "请勿在展厅使用闪光灯。",
        "午后分组讨论三件展品。", "Write down one question per gallery.",
        "离馆前交回语音导览设备。", "备注：请保留纸质参观凭证。",
    )),
    ("社区植物调查", "Neighborhood Plant Survey", (
        "沿河道选择三个观测点。", "用简图标记树冠和花坛的位置。", "逐条记录叶形、花色和株高。",
        "Sample ID P-08 is near the bridge.", "Record the leaf width in centimeters.", "遇到不熟悉的植物先拍照。",
        "傍晚回到教室核对编号。", "Compare the sunny and shaded plots.",
        "同一株植物不重复计数。", "备注：铅笔所写的边注字迹较浅。",
    )),
)


def render_page(document: tuple, font_path: Path) -> tuple[np.ndarray, list[np.ndarray], str]:
    title, english, body = document
    page = Image.new("RGB", PAGE_SIZE, (247, 247, 243))
    draw = ImageDraw.Draw(page)
    specs = [("合成样例 / SYNTHETIC", 28, 62, 83), (title, 49, 121, 24), (english, 31, 198, 31)]
    for index, text in enumerate(body):
        size = 28 if index < 8 else 22 if index == 8 else 19
        gray = 36 if index < 8 else 114 if index == 8 else 154
        specs.append((text, size, 307 + 77 * index, gray))
    boxes = []
    for text, size, y, gray in specs:
        font = sample_font(font_path, size)
        left, top, right, bottom = draw.textbbox((76, y), text, font=font)
        if right >= PAGE_SIZE[0] - 30 or bottom >= PAGE_SIZE[1] - 30:
            raise DocscanError("sample_layout_failed", f"Text does not fit the synthetic page: {text!r}")
        draw.text((76, y), text, font=font, fill=(gray, gray, gray))
        boxes.append(np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32))
    return cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR), boxes, "\n".join(text for text, *_ in specs) + "\n"


def photograph(page: np.ndarray, condition: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    width, height = PHOTO_SIZE
    if condition.startswith("tilt"):
        destination = [[195, 74], [922, 221], [1020, 1277], [89, 1350]]
    else:
        destination = [[126, 106], [961, 96], [971, 1295], [133, 1303]]
    points = (np.asarray(destination) + rng.integers(-12, 13, size=(4, 2))).astype(np.float32)
    pw, ph = PAGE_SIZE
    source = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    transform = cv2.getPerspectiveTransform(source, points)
    warped = cv2.warpPerspective(page, transform, PHOTO_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), transform, PHOTO_SIZE).astype(np.float32) / 255
    x, y = np.meshgrid(np.linspace(0, 1, width, dtype=np.float32), np.linspace(0, 1, height, dtype=np.float32))
    background = np.empty((height, width, 3), dtype=np.float32)
    background[:] = (72, 86, 81)
    background += (8 * np.sin(19 * y) + 11 * x)[:, :, None]
    gradient = 0.98 + 0.045 * x - 0.045 * y
    parameters = {"gradient": [0.98, 0.045, -0.045], "noise_sigma": 1.25, "shadow": None}
    if condition.endswith("shadow"):
        strength = float(rng.uniform(0.45, 0.68))
        edge = float(rng.uniform(0.45, 0.58))
        diagonal = 0.62 * x + 0.46 * y
        shadow = 0.5 * (1 + np.tanh((diagonal - edge) / 0.027))
        gradient = (1.02 - 0.12 * x - 0.15 * y) * (1 - strength * shadow)
        parameters.update(gradient=[1.02, -0.12, -0.15],
                          shadow={"strength": strength, "edge": edge, "softness": 0.027, "direction": [0.62, 0.46]})
    canvas = warped * (mask * gradient)[:, :, None] + background * (1 - mask)[:, :, None]
    canvas += rng.normal(0, parameters["noise_sigma"], (height, width)).astype(np.float32)[:, :, None]
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8), points, transform, parameters


def make_samples(output: Path, seed: int, font: Path | None = None) -> dict:
    if not 0 <= seed < 2**63:
        raise DocscanError("invalid_seed", "Seed must be in [0, 2**63)")
    font_path = chinese_font(font)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DocscanError("sample_output_exists", f"Use an empty directory for samples: {output}")
    rng = np.random.default_rng(seed)
    manifest = []
    font_info = {"file": font_path.name, "sha256": file_sha256(font_path),
                 "family_style": sample_font(font_path, 28).getname(), "variation_preference": "Regular"}
    for index, document in enumerate(DOCUMENTS, start=1):
        doc_id = f"syn-{index:02}"
        split = "dev" if index <= 2 else "test"
        page, text_boxes, text = render_page(document, font_path)
        gt_path = f"{doc_id}.txt"
        (output / gt_path).write_text(text, encoding="utf-8")
        for condition in CONDITIONS:
            image_id = f"{doc_id}-{condition}"
            stem = "sample" if index == 1 and condition == "tilt_shadow" else image_id
            photo, corners, transform, lighting = photograph(page, condition, rng)
            Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)).save(output / f"{stem}.jpg", quality=94, subsampling=0)
            polygons = cv2.perspectiveTransform(np.asarray(text_boxes), transform)
            annotation = {
                "source": "synthetic", "coordinate_space": NORMALIZED_SPACE, "image_size": PHOTO_SIZE,
                "points": corners.tolist(), "order": CORNER_ORDER, "text_polygons": polygons.tolist(),
                "seed": seed, "font": font_info, "page_size": PAGE_SIZE,
                "page_to_photo": transform.tolist(), "lighting": lighting,
            }
            corners_name = f"{stem}.corners.json"
            write_json(output / corners_name, annotation)
            manifest.append({"image_id": image_id, "doc_id": doc_id, "image_path": f"{stem}.jpg",
                             "gt_text_path": gt_path, "corners_path": corners_name, "split": split,
                             "condition": condition, "source": "synthetic", "page_orientation": "portrait",
                             "notes": "Program-generated print and photo simulation; no real photo"})
    write_csv(output / "manifest.csv", MANIFEST_FIELDS, manifest)
    write_json(output / "dataset.json", {"source": "synthetic", "seed": seed, "font": font_info,
                                         "images": [row["image_path"] for row in manifest]})
    return {"status": "ok", "source": "synthetic", "manifest": str(output / "manifest.csv"),
            "seed": seed, "images": len(manifest), "documents": len(DOCUMENTS), "font": font_info}
