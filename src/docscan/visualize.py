from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .common import CONDITIONS, METHODS, DocscanError, file_sha256, read_csv, write_csv, write_json

METHOD_LABELS = {"A": "原始照片", "B": "矫正灰度", "C": "全局 Otsu", "D": "自适应阈值",
                 "E": "光照校正灰度", "F": "校正后自适应阈值"}
CONDITION_LABELS = {"ALL": "全部", "front_even": "正拍均光", "tilt_even": "斜拍均光",
                    "front_shadow": "正拍阴影", "tilt_shadow": "斜拍阴影"}
CONTRASTS = (("E", "B"), ("F", "D"), ("C", "B"), ("D", "B"), ("F", "E"), ("B", "A"))
CASE_LABELS = {"improved": "改善", "unchanged": "持平", "degraded": "退化"}


def read_run(path: Path, mode: str) -> dict:
    try:
        data = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if data["command"] != "benchmark" or data["mode"] != mode:
            raise DocscanError("wrong_benchmark_mode", f"Expected {mode} benchmark: {path}")
        prefix = "auto_" if mode == "auto" else ""
        for output in ("results", "summary"):
            digest = data.get(f"{output}_sha256")
            if digest and file_sha256(path / f"{prefix}{output}.csv") != digest:
                raise DocscanError("benchmark_artifact_changed", f"{prefix}{output}.csv differs from {path}/run.json")
        return data
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise DocscanError("invalid_benchmark_run", f"Cannot read {path}/run.json: {error}") from error


def _scope(run: dict) -> str:
    source = "合成验证" if run["source"] == "synthetic" else "真实照片"
    split = "开发集" if run["split"] == "dev" else "测试集"
    return f"{source} · {split}"


def _figure_style(font: Path | None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    from .fonts import chinese_font

    font_path = chinese_font(font, prefer_static=True)
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "axes.titleweight": "bold", "savefig.facecolor": "white"})
    return plt, {"file": font_path.name, "sha256": file_sha256(font_path), "family": family}


def _plot_overview(plt, run: dict, summary: list[dict], path: Path) -> str:
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    overall = [row for row in summary if row["condition"] == "ALL"]
    if not overall:
        return ""
    figure, ax = plt.subplots(figsize=(10.0, 4.4), layout="constrained")
    colors = ("#6a7685", "#93a9ba", "#ba7850", "#8e70a7", "#227c94", "#3a8a67")
    expected = int(overall[0]["expected"])
    if run["status"] == "ok" and all(int(row["complete_pairs"]) == expected for row in overall):
        title = f"{_scope(run)}   A–F 字符错误率"
        caption = "微平均 CER  越低越好"
        present = [condition for condition in ("ALL", *CONDITIONS) if any(row["condition"] == condition for row in summary)]
        lookup = {(row["condition"], row["method"]): row for row in summary}
        x = np.arange(len(present))
        for index, (method, color) in enumerate(zip(METHODS, colors, strict=True)):
            values = [float(lookup[(condition, method)]["micro_cer"]) for condition in present]
            ax.bar(x + (index - 2.5) * 0.125, values, width=0.113, color=color, label=method)
        ax.set_xticks(x, [f"{CONDITION_LABELS[condition]}\n(n={lookup[(condition, 'A')]['expected']})" for condition in present])
        ax.legend(title="方法", ncols=6, frameon=False, loc="upper right")
        ax.set_ylim(0, max(0.02, ax.get_ylim()[1] * 1.15))
        filename = "cer_by_condition.png"
    else:
        title = f"{_scope(run)}   实验覆盖率"
        caption = "有效 OCR 记录占比"
        ax.bar([row["method"] for row in overall], [float(row["coverage"]) for row in overall], color=colors)
        ax.set_ylim(0, 1.08)
        ax.text(0.99, 0.96, f"完整配对 {overall[0]['complete_pairs']} / {expected}", transform=ax.transAxes, ha="right", va="top")
        filename = "coverage.png"
    ax.set_title(title, loc="left", pad=14)
    ax.set_ylabel(caption)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22)
    figure.savefig(path / filename, dpi=165)
    plt.close(figure)
    return filename


def _find_cases(groups: dict) -> tuple[list[dict], dict]:
    counts = {f"{target}/{base}": {key: 0 for key in CASE_LABELS} for target, base in CONTRASTS}
    picked = {}
    for image_id, methods in sorted(groups.items()):
        if any(method not in methods or methods[method]["status"] not in {"ok", "no_text"} for method in METHODS):
            continue
        for target, base in CONTRASTS:
            lhs, rhs = float(methods[target]["cer"]), float(methods[base]["cer"])
            delta = lhs - rhs
            label = "improved" if delta < 0 else "degraded" if delta > 0 else "unchanged"
            counts[f"{target}/{base}"][label] += 1
            if label not in picked or abs(delta) > abs(picked[label]["delta"]):
                picked[label] = {"image_id": image_id, "condition": methods["A"]["condition"], "kind": label,
                                 "target": target, "baseline": base, "target_cer": lhs, "baseline_cer": rhs, "delta": delta}
    return [picked[key] for key in CASE_LABELS if key in picked], counts


def _crop_boxes(run_dir: Path, image_id: str, methods: dict, size: tuple[int, int]) -> dict:
    import cv2
    import numpy as np

    w, h = size
    geometry = json.loads((run_dir / "pages" / image_id / "geometry.json").read_text(encoding="utf-8"))
    polygons = []
    for method in ("B", "E", "F"):
        filename = methods[method]["ocr_json"]
        if filename:
            ocr = json.loads((run_dir / filename).read_text(encoding="utf-8"))
            polygons = [line["polygon"] for line in ocr["lines"][-2:]]
        if polygons:
            break
    if polygons:
        points = np.array(polygons).reshape(-1, 2)
        low = np.floor(points.min(axis=0) - [0.035 * w, 0.018 * h]).astype(int)
        high = np.ceil(points.max(axis=0) + [0.035 * w, 0.018 * h]).astype(int)
        left, top = np.maximum(low, [0, 0])
        right, bottom = np.minimum(high, [w, h])
    else:
        left, top, right, bottom = round(w * 0.08), round(h * 0.66), round(w * 0.84), round(h * 0.84)
    region = np.float32([[[left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1]]])
    raw = cv2.perspectiveTransform(region, np.asarray(geometry["rectified_to_input"]))[0]
    x, y, ww, hh = cv2.boundingRect(raw)
    return {"A": (max(x, 0), max(y, 0), x + ww, y + hh), "B-F": (left, top, right, bottom)}


def _plot_cases(plt, run_dir: Path, path: Path, run: dict, images: dict, chosen: set[str]) -> dict:
    import numpy as np
    from PIL import Image

    figures = {}
    for image_id in sorted(chosen):
        methods = images[image_id]
        full, crops = {}, {}
        for method in METHODS:
            with Image.open(run_dir / methods[method]["variant_path"]) as source:
                full[method] = np.asarray(source.convert("RGB"))
        h, w = full["B"].shape[:2]
        boxes = _crop_boxes(run_dir, image_id, methods, (w, h))
        for method in METHODS:
            x0, y0, x1, y1 = boxes["A" if method == "A" else "B-F"]
            crops[method] = full[method][y0:y1, x0:x1]
        figure_paths = {}
        for label, panels, size, caption in (("page", full, (10, 8), "全页对照"), ("crop", crops, (10, 4.5), "同一区域文字放大")):
            figure, axes = plt.subplots(2, 3, figsize=size, layout="constrained")
            figure.suptitle(f"{_scope(run)} · {image_id} · {caption}", fontsize=14)
            for method, ax in zip(METHODS, axes.flat, strict=True):
                ax.imshow(panels[method], interpolation="nearest")
                ax.set_title(f"{method}  {METHOD_LABELS[method]}", fontsize=11)
                ax.axis("off")
            name = f"{label}-{image_id}.png"
            figure.savefig(path / name, dpi=156)
            plt.close(figure)
            figure_paths[label] = name
        figure_paths["binary_ink_fraction"] = {method: float(np.mean(crops[method][:, :, 0] == 0)) for method in "CDF"}
        figures[image_id] = figure_paths
    return figures


def _plot_lighting(plt, run_dir: Path, path: Path, run: dict, groups: dict) -> str:
    from PIL import Image

    rows = []
    for condition in CONDITIONS:
        for image_id, group in sorted(groups.items()):
            if group["A"]["condition"] == condition and all(group[method]["variant_path"] for method in "BEF"):
                background = run_dir / "pages" / image_id / "illumination_map.png"
                if background.is_file():
                    rows.append((condition, group, background))
                    break
    if not rows:
        return ""
    figure, axes = plt.subplots(len(rows), 4, figsize=(9.3, 2.7 * len(rows)), squeeze=False, layout="constrained")
    figure.suptitle(f"{_scope(run)}   四种拍摄条件的光照校正", fontsize=15)
    for (condition, group, background), row in zip(rows, axes, strict=True):
        paths = [run_dir / group["B"]["variant_path"], background,
                 run_dir / group["E"]["variant_path"], run_dir / group["F"]["variant_path"]]
        for index, (image_path, ax, title) in enumerate(zip(paths, row, ("B 矫正灰度", "背景亮度", "E 校正灰度", "F 二值图"), strict=True)):
            with Image.open(image_path) as source:
                ax.imshow(source.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{CONDITION_LABELS[condition]} · {title}" if index == 0 else title, fontsize=10)
            ax.axis("off")
    figure.savefig(path / "lighting.png", dpi=155)
    plt.close(figure)
    return "lighting.png"


def summarize(run_dir: Path, font: Path | None = None) -> dict:
    run = read_run(run_dir, "shared-corners")
    results_file, summary_file = run_dir / "results.csv", run_dir / "summary.csv"
    results, summary = read_csv(results_file), read_csv(summary_file)
    for row in results + summary:
        if (row["run_id"], row["source"], row["split"]) != (run["run_id"], run["source"], run["split"]):
            raise DocscanError("run_provenance_mismatch", f"CSV row does not belong to {run['run_id']}")
    path = run_dir / "analysis"
    path.mkdir(exist_ok=True)
    index = {"run_id": run["run_id"], "source": run["source"], "split": run["split"],
             "results_sha256": file_sha256(results_file), "summary_sha256": file_sha256(summary_file),
             "status": run["status"], "figures": {}, "cases": [], "contrasts": {}}
    columns = ("run_id", "source", "split", "condition", "method", "status", "measured", "expected",
               "mean_preprocess_ms", "mean_ocr_ms", "mean_total_ms", "median_total_ms")
    write_csv(path / "timing.csv", columns, [{key: row[key] for key in columns} for row in summary])
    if results:
        plt, index["font"] = _figure_style(font)
        groups = defaultdict(dict)
        for row in results:
            groups[row["image_id"]][row["method"]] = row
        cases, contrasts = _find_cases(groups)
        chosen = {case["image_id"] for case in cases}
        for condition in ("front_even", "tilt_shadow"):
            choices = sorted(image_id for image_id, group in groups.items() if group["A"]["condition"] == condition and
                             all(group[method]["variant_path"] and group[method]["status"] in {"ok", "no_text"} for method in METHODS))
            if choices:
                chosen.add(choices[0])
        index.update(cases=cases, contrasts=contrasts, case_figures=_plot_cases(plt, run_dir, path, run, groups, chosen))
        index["figures"] = {"overview": _plot_overview(plt, run, summary, path),
                            "lighting": _plot_lighting(plt, run_dir, path, run, groups)}
    write_json(path / "index.json", index)
    return {"analysis_dir": str(path), **index}
