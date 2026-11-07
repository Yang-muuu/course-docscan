from __future__ import annotations

import json
import shutil
from pathlib import Path

from .common import CONDITIONS, DocscanError, file_sha256, new_run, read_csv, write_json
from .visualize import CASE_LABELS, CONDITION_LABELS, METHOD_LABELS, read_run, summarize

TITLE = "手机拍摄文档的透视矫正 光照校正与文字识别"
REFERENCES = (
    ("OpenCV 几何变换", "https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html"),
    ("OpenCV 形态学操作", "https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html"),
    ("OpenCV 图像阈值", "https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html"),
    ("PaddleOCR 3.3.1 通用 OCR", "https://www.paddleocr.ai/v3.3.1/en/version3.x/pipeline_usage/OCR.html"),
    ("PyMuPDF TextWriter 文字层", "https://pymupdf.readthedocs.io/en/latest/textwriter.html"),
)


def _percentage(value: str | float | None) -> str:
    return f"{float(value):.2%}" if value not in (None, "") else "—"


def _time(value: str | float | None) -> str:
    return f"{float(value):.1f}" if value not in (None, "") else "—"


def _results_blocks(run: dict, overall: list[dict], analysis: dict) -> list[tuple[str, object]]:
    if not overall:
        return [("p", f"本次运行没有可分析的配对记录。运行状态为 {run['status']}，原因见 run.json。")]
    complete = run["status"] == "ok" and all(row["complete_pairs"] == row["expected"] for row in overall)
    rows = [[row["method"], f"{row['measured']}/{row['expected']}", row["S"], row["D"], row["I"], row["N"],
             _percentage(row["micro_cer"]) if complete else "—"] for row in overall]
    pairing = f"完整配对 {overall[0]['complete_pairs']}/{overall[0]['expected']} 张。"
    if complete:
        best = min(float(row["micro_cer"]) for row in overall)
        methods = "、".join(row["method"] for row in overall if float(row["micro_cer"]) == best)
        observation = f"本批微平均 CER 最低的方法为 {methods}，实测为 {_percentage(best)}。"
    else:
        observation = "本次实验有缺失项。表中的错误计数只覆盖各方法实际完成的样本，整体 CER 比较留待补齐配对后进行。"
    blocks = [("p", pairing + observation),
              ("table", (["方法", "有效/应有", "替换 S", "删除 D", "插入 I", "真值 N", "CER"], rows)),
              ("caption", "表 1  逐组错误计数与按字符数加权的 CER")]
    overview = analysis["figures"].get("overview")
    if overview:
        blocks.append(("image", (overview, "图 1  各拍摄条件的微平均 CER" if complete else "图 1  各方法的有效记录覆盖率")))
    timing = [[row["method"], row["measured"], _time(row["mean_preprocess_ms"]), _time(row["mean_ocr_ms"]),
               _time(row["mean_total_ms"]), _time(row["median_total_ms"])] for row in overall]
    blocks.extend([("h2", "耗时"), ("table", (["方法", "有效", "前处理均值", "推理均值", "总时均值", "总时中位数"], timing)),
                   ("caption", "表 2  有效记录耗时，单位 ms。总时为前处理与推理之和。"),
                   ("p", "前处理包含 EXIF 解码、对应方法的图像变换及 OCR 输入缩放。B 至 F 共用一次矫正和灰度计算，表内每个方法计入完整的共享前处理时间。模型下载、加载、预热、写盘和 CER 计算另计。")])
    load_ms = run.get("ocr", {}).get("model_load_ms")
    attempts = run.get("warmup", {}).get("attempts", [])
    if load_ms is not None and attempts:
        warm_ms = sum(attempt.get("predict_ms") or 0 for attempt in attempts)
        blocks.append(("p", f"本次模型加载 {_time(load_ms)} ms，预热推理合计 {_time(warm_ms)} ms。"))
    return blocks


def _case_blocks(analysis: dict) -> list[tuple[str, object]]:
    blocks = [("h2", "逐图变化与失败案例")]
    changes = []
    for name in ("E/B", "F/D", "F/E"):
        if name in analysis["contrasts"]:
            count = analysis["contrasts"][name]
            changes.append(f"{name} 有 {count['improved']} 张改善、{count['unchanged']} 张持平、{count['degraded']} 张退化。")
    if changes:
        blocks.append(("p", " ".join(changes)))
    cases = {case["kind"]: case for case in analysis["cases"]}
    for kind, label in CASE_LABELS.items():
        case = cases.get(kind)
        if case is None:
            blocks.append(("p", f"本次完整配对中没有观察到{label}案例。"))
            continue
        blocks.append(("p", f"{label}案例 {case['image_id']}（{CONDITION_LABELS[case['condition']]}）。"
                      f"{case['baseline']} 的 CER 为 {_percentage(case['baseline_cer'])}，"
                      f"{case['target']} 为 {_percentage(case['target_cer'])}。"))
    example = cases.get("degraded") or cases.get("improved") or cases.get("unchanged")
    if example:
        image_id = example["image_id"]
        figures = analysis["case_figures"][image_id]
        blocks.append(("image", (figures["crop"], f"图 2  {image_id} 的同一区域文字放大。A 的裁区由共享透视矩阵逆映射。")))
        if example["target"] in figures["binary_ink_fraction"]:
            fraction = figures["binary_ink_fraction"][example["target"]]
            detail = f"图 2 中 {example['target']} 的局部黑色像素占 {_percentage(fraction)}。"
            if example["target"] == "C" and fraction >= 0.7 and example["condition"].endswith("shadow"):
                detail += "全局阈值将大片阴影纸面归为前景，页尾文字与黑色背景相连。"
            blocks.append(("p", detail))
    blocks.append(("p", "全局阈值会把低亮度纸面与笔画一起判为黑色。自适应阈值的常数过大时，阈值下移，浅笔画容易变白。背景估计窗口过小会把较粗笔画吸收到亮度图，窗口过大则可能跨越阴影边界。四角偏差会裁掉边缘文字，需结合叠加图检查。"))
    return blocks


def _auto_blocks(run: dict, auto_dir: Path | None) -> list[tuple[str, object]]:
    if auto_dir is None:
        return []
    auto = read_run(auto_dir, "auto")
    keys = ("source", "split", "manifest_sha256")
    if any(auto.get(key) != run.get(key) for key in keys):
        raise DocscanError("auto_run_mismatch", "Automatic evaluation must use the same source, split and manifest")
    overall = next((row for row in read_csv(auto_dir / "auto_summary.csv") if row["condition"] == "ALL"), None)
    if overall is None:
        return [("h2", "独立自动定位评价"), ("p", f"运行 {auto['run_id']} 没有生成有效自动定位评价。")]
    config = auto["effective_config"]["evaluation"]
    return [("h2", "独立自动定位评价"),
            ("p", f"运行 {auto['run_id']} 在 {overall['expected']} 张中评价了 {overall['evaluated']} 张，"
             f"得到候选 {overall['candidates']} 张，几何通过 {overall['geometry_passed']} 张。"
             f"几何阈值为 IoU ≥ {config['auto_iou_min']:.2f}，最大角点误差 ≤ 图像对角线的 {_percentage(config['auto_corner_error_max'])}。"),
            ("p", f"{overall['text_annotated']} 张带文字区域标注，其中 {overall['auto_verified']} 张同时通过几何与文字保留检查。"
             f"四角标注补救成功 {overall['manual_rescues']} 张，计数与自动通过分开。缺少文字区域标注时只报告几何结果。")]


def _blocks(run: dict, summary: list[dict], analysis: dict, auto_dir: Path | None) -> tuple[str, list[tuple[str, object]]]:
    synthetic = run["source"] == "synthetic"
    split = "开发集" if run["split"] == "dev" else "测试集"
    subtitle = "课程项目报告  合成验证  真实实验待补" if synthetic else f"课程项目报告  真实照片{split}"
    data = run.get("dataset", {})
    count = data.get("images", 0)
    source = "合成图像" if synthetic else "真实照片"
    start = (f"项目将手机文档照片矫正为正视图，比较光照校正和阈值处理对预训练 OCR 字符错误率的影响。"
             f"本稿对应{split}运行 {run['run_id']}，选择 {data.get('documents', 0)} 份文档的 {count} 张{source}。")
    if synthetic:
        start += "结果描述程序生成的光照和透视条件，真实手机拍摄实验待补。"
    if data.get("status") == "partial_dataset":
        start += "真实数据规模低于课程计划，以下结果对应本批样本。"
    conditions = data.get("conditions", {})
    data_conditions = "、".join(f"{CONDITION_LABELS[key]} {conditions[key]} 张" for key in CONDITIONS if key in conditions)
    config = run["effective_config"]
    enh, ocr, geo = config["enhance"], config["ocr"], config["geometry"]
    engine = run.get("ocr", {})
    models = engine.get("model_names", {})
    model_sentence = (f"OCR 运行使用 PaddleOCR {engine['engine_version']} 和"
                      f" {models.get('detection')}、{models.get('recognition')}，设备为 {ocr['device']}。"
                      if engine.get("engine_version") else "OCR 运行环境详见对应 run.json。")
    blocks = [("p", start), ("h1", "1 GitHub 链接"), ("p", "课程仓库链接待填写。"),
              ("h1", "2 任务与算法"),
              ("p", "图像先按 EXIF 方向归一化，再通过四角与目标矩形之间的单应矩阵完成透视矫正。自动定位在缩小图上筛选凸四边形，用面积、边界支持和形状评分，再将顶点回映至原图。可搜索 PDF 把扫描图作为底图，复用同一布局中的 OCR 文本框与隐藏中文文字层。"),
              ("p", f"背景估计采用灰度形态学闭运算与高斯平滑。核宽按短边的 {enh['background_scale']:.3f} 取奇数，再约束在 {enh['background_min']} 至 {enh['background_max']} 像素。"
               f"校正把输入灰度乘以 {enh['target_white']:.0f}，除以背景估计与 {enh['background_floor']:.0f} 的较大值，然后四舍五入并裁剪到 0 至 255。"
               f"自适应高斯阈值窗口为 {enh['adaptive_block']}，常数为 {enh['adaptive_c']:.0f}。"),
              ("p", model_sentence + f"推理输入最长边限制为 {ocr['max_input_side']} 像素。页面方向、文档展平和文字行方向模块均关闭。传统图像处理负责几何与光照，预训练模型负责检测和识别。"),
              ("h1", "3 数据与协议"),
              ("p", (f"本批条件分布为 {data_conditions}。" if data_conditions else "本批尚无可核对的拍摄条件记录。") +
               " 按文档内容隔离开发与测试。B 至 F 使用相同标注四角、输出尺寸和插值。自动定位另行评价。"),
              ("p", f"页面规格为 {geo['page_format']}，矫正最长边上限 {geo['max_long_edge']} 像素。CER 先对真值和预测做 NFKC 和空白去除，保留大小写与标点，再用编辑对齐求替换 S、删除 D、插入 I。总体 CER 为累计 (S+D+I)/N。空预测计全部删除，运行异常单列。"),
              ("h1", "4 代码与处理分组"),
              ("p", "项目包含图像读写、四角与单应、光照处理、OCR 适配、PDF 导出、数据验证、评价和报告构建。CLI 支持单张或目录，保存中间图、原始模型结果、排序文本、参数、文件哈希和状态。"),
              ("table", (["组别", "OCR 输入"], [[key, value] for key, value in METHOD_LABELS.items()])),
              ("caption", "A 与 B 同时改变透视几何和灰度。C、D、E 分别对比 B。F 相对 D 检查光照校正增量，相对 E 检查二值化增量。")]
    if run.get("freeze"):
        freeze = run["freeze"]
        blocks.append(("p", f"测试配置在开发运行 {freeze['dev_run_id']} 后冻结。{freeze['selection_note']}"))
    blocks.append(("h1", "5 结果对比"))
    blocks.extend(_results_blocks(run, summary, analysis))
    if summary:
        blocks.extend(_case_blocks(analysis))
    blocks.extend(_auto_blocks(run, auto_dir))
    blocks.extend([("h1", "6 局限与结论"),
                   ("p", "本流程针对平整、正向、单栏印刷页。局部反光、强阴影、纸面弯曲和复杂布局可能影响定位和识别。"
                    + ("合成测试用于验证实现和比较流程。真实数据补齐后需在真实开发集重新选参，在冻结条件下独立评价真实测试集。"
                       if synthetic else "结论适用于本批文档、拍摄条件及固定推理配置。")),
                   ("p", "结果对应本报告旁的 summary.csv、timing.csv 及 evidence.json，逐图错误计数和状态见对应运行的 results.csv。"),
                   ("h1", "参考资料"), *[("link", (title, url)) for title, url in REFERENCES]])
    return subtitle, blocks


def _markdown(subtitle: str, blocks: list[tuple[str, object]]) -> str:
    lines = [f"# {TITLE}", "", subtitle, ""]
    for kind, value in blocks:
        if kind in {"p", "caption"}:
            lines.append(value)
        elif kind in {"h1", "h2"}:
            lines.append(f"{'##' if kind == 'h1' else '###'} {value}")
        elif kind == "table":
            headings, rows = value
            for row in (headings, ["---"] * len(headings), *rows):
                lines.append("| " + " | ".join(row) + " |")
        elif kind == "image":
            name, caption = value
            lines.extend((f"![{caption}](figures/{name})", "", caption))
        elif kind == "link":
            lines.append(f"[{value[0]}]({value[1]})")
        lines.append("")
    return "\n".join(lines)


def build_report(run_dir: Path, output: Path, auto_dir: Path | None = None) -> dict:
    from .report_docx import write_docx

    run = read_run(run_dir, "shared-corners")
    analysis_path = run_dir / "analysis" / "index.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else summarize(run_dir)
    expected = {"run_id": run["run_id"], "results_sha256": file_sha256(run_dir / "results.csv"),
                "summary_sha256": file_sha256(run_dir / "summary.csv")}
    if any(analysis.get(key) != value for key, value in expected.items()):
        raise DocscanError("stale_analysis", "Regenerate analysis with summarize before building the report")
    overall = [row for row in read_csv(run_dir / "summary.csv") if row["condition"] == "ALL"]
    subtitle, blocks = _blocks(run, overall, analysis, auto_dir)
    path, _ = new_run(output / run["run_id"])
    figures = path / "figures"
    figures.mkdir()
    for kind, value in blocks:
        if kind == "image":
            shutil.copyfile(analysis_path.parent / value[0], figures / value[0])
    for file in (run_dir / "summary.csv", analysis_path.parent / "timing.csv"):
        shutil.copyfile(file, path / file.name)
    if auto_dir is not None:
        shutil.copyfile(auto_dir / "auto_summary.csv", path / "auto_summary.csv")
    (path / "report.md").write_text(_markdown(subtitle, blocks), encoding="utf-8")
    write_docx(path / "report.docx", TITLE, subtitle, blocks, figures)
    write_json(path / "evidence.json", {**expected, "source": run["source"], "split": run["split"],
                                         "config_sha256": run["config_sha256"], "benchmark_status": run["status"],
                                         "auto_run_id": read_run(auto_dir, "auto")["run_id"] if auto_dir else None,
                                         "case_records": analysis["cases"]})
    return {"status": "ok", "report_dir": str(path), "run_id": run["run_id"], "source": run["source"],
            "benchmark_status": run["status"]}
