from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from .common import METHODS, DocscanError, dependency_versions, elapsed_ms, error_record, finish_run, hardware, new_run, run_metadata


def show(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def doctor(args: argparse.Namespace) -> int:
    from .config import load_config
    from .fonts import chinese_font, sample_font
    from .models import DEWARP_MODELS, model_status

    config, _ = load_config(args.config)
    versions = dependency_versions()
    report = {
        "python": sys.version.split()[0], "hardware": hardware(), "dependencies": versions,
        "device": config["ocr"]["device"], "models": model_status(Path(config["ocr"]["model_dir"])),
        "dewarp_model": model_status(Path(config["ocr"]["model_dir"]), DEWARP_MODELS)["dewarp"],
        "note": "This command checks local prerequisites; use ocr-check to verify inference.",
    }
    try:
        font = chinese_font(args.font)
        report["sample_font"] = {"path": str(font), "family": sample_font(font, 24).getname()}
    except (OSError, DocscanError) as error:
        report["sample_font"] = error_record(error)
    try:
        import pymupdf

        font = pymupdf.Font(fontname="cjk")
        report["pdf"] = {"font": font.name, "chinese_glyphs_present": all(font.has_glyph(ord(c)) for c in "文档中文")}
    except (ImportError, RuntimeError, ValueError) as error:
        report["pdf"] = error_record(error)
    show(report)
    return 0


def prepare(args: argparse.Namespace) -> int:
    from .config import load_config
    from .models import prepare_models

    config, _ = load_config(args.config)
    result = prepare_models(Path(config["ocr"]["model_dir"]), dewarp=args.dewarp)
    show(result)
    return int(result["status"] != "ok")


def ocr_check(args: argparse.Namespace) -> int:
    from .config import load_config
    from .dataset import identified_source
    from .image_io import NORMALIZED_SPACE, load_photo
    from .ocr import PaddleEngine, download_record, save_ocr

    config, _ = load_config(args.config)
    path, run_id = new_run(args.output)
    run = run_metadata(run_id, config, args.source)
    run.update(command="ocr-check", input=str(args.input), status="error")
    start = perf_counter()
    try:
        run["source"] = identified_source(args.input, args.source)
        photo = load_photo(args.input)
        run.update(input_sha256=photo.sha256, input_size=photo.size)
        engine = PaddleEngine(config["ocr"], reading_order=args.reading_order)
        result = engine.recognize(photo.bgr, NORMALIZED_SPACE)
        run["outputs"] = save_ocr(path, result)
        run.update(status=result["status"], ocr=engine.metadata(), model_preparation=download_record(engine))
        run["elapsed_ms"]["predict"] = result["elapsed_ms"]
        if result["status"] == "error":
            run.update(error_code=result["error_code"], error=result["error"])
    except (OSError, DocscanError) as error:
        run.update(**error_record(error))
    finally:
        run["elapsed_ms"]["total"] = elapsed_ms(start)
        finish_run(path, run)
    show({"run_dir": str(path), "status": run["status"], "error": run.get("error")})
    return int(run["status"] != "ok")


def make_samples(args: argparse.Namespace) -> int:
    from .samples import make_samples as generate

    show(generate(args.output, args.seed, args.font))
    return 0


def validate_data(args: argparse.Namespace) -> int:
    from .dataset import validate_data as validate

    report = validate(args.manifest, require_real_plan=args.require_real_plan)
    show(report)
    return int(report["status"] != "ok")


def scan(args: argparse.Namespace) -> int:
    from .config import load_config, validate_config
    from .pipeline import scan as run_scan

    config, _ = load_config(args.config)
    for section, key, value in (("geometry", "page_format", args.page_format),
                                ("geometry", "max_long_edge", args.max_long_edge),
                                ("pdf", "font", str(args.pdf_font) if args.pdf_font else None)):
        if value is not None:
            config[section][key] = value
    validate_config(config)
    options = {key: getattr(args, key) for key in ("method", "ocr", "pdf_mode", "source", "page_orientation", "auto_first",
                                                    "reading_order")}
    result = run_scan(args.input, args.corners, args.output, config, options)
    show(result)
    return int(result["status"] != "ok")


def _gutter(value: str) -> tuple[float, float]:
    try:
        parts = [float(item) for item in value.split(",")]
        if len(parts) not in (1, 2) or not all(0.05 <= item <= 0.95 for item in parts):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use one x fraction or top,bottom fractions, each from 0.05 to 0.95") from error
    return parts[0], parts[-1]


def book(args: argparse.Namespace) -> int:
    from .book import convert_book
    from .config import load_config

    config, _ = load_config(args.config)
    if args.pdf_font is not None:
        config["pdf"]["font"] = str(args.pdf_font)
    options = {key: getattr(args, key) for key in ("layout", "gutter", "page_format", "page_orientation",
                                                    "pdf_mode", "dewarp", "source", "dpi", "same_page", "reading_order")}
    result = convert_book(args.input, args.output, args.corners, config, options)
    show(result)
    return int(result["status"] != "ok")


def verify_pdf(args: argparse.Namespace) -> int:
    from .export_pdf import verify_pdf as verify

    expected = args.expected_text.read_text(encoding="utf-8-sig")
    show(verify(args.input, expected, args.keyword, args.ocr_json, args.background, args.render, page_number=args.page))
    return 0


def benchmark(args: argparse.Namespace) -> int:
    from .config import load_config
    from .evaluate import benchmark as run_benchmark

    config, freeze = load_config(args.config)
    report = run_benchmark(args.manifest, args.output, args.source, args.split, args.mode, config, freeze)
    show(report)
    return int(report["status"] != "ok")


def freeze_config(args: argparse.Namespace) -> int:
    from .evaluate import freeze_config as freeze

    show(freeze(args.dev_run, args.output, args.note))
    return 0


def summarize(args: argparse.Namespace) -> int:
    from .visualize import summarize as make_summary

    report = make_summary(args.run_dir, args.font)
    show({key: report[key] for key in ("analysis_dir", "run_id", "source", "status", "figures")})
    return 0


def build_report(args: argparse.Namespace) -> int:
    from .report import build_report as build

    show(build(args.run_dir, args.output, args.auto_run_dir))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="docscan", description="Scan document photos and evaluate OCR preprocessing.")
    commands = root.add_subparsers(required=True, dest="command")
    check = commands.add_parser("doctor", help="Inspect dependencies, local weights and fonts without downloading.")
    check.add_argument("--config", type=Path)
    check.add_argument("--font", type=Path)
    check.set_defaults(handler=doctor)
    models = commands.add_parser("prepare-models", help="Download and verify the two pinned PP-OCRv5 weights.")
    models.add_argument("--config", type=Path)
    models.add_argument("--dewarp", action="store_true", help="Prepare the optional UVDoc weights instead.")
    models.set_defaults(handler=prepare)
    ocr = commands.add_parser("ocr-check", help="Run real PaddleOCR inference directly on one image.")
    ocr.add_argument("--input", type=Path, required=True)
    ocr.add_argument("--output", type=Path, required=True)
    ocr.add_argument("--config", type=Path)
    ocr.add_argument("--source", choices=("real", "synthetic", "unspecified"), default="unspecified")
    ocr.set_defaults(handler=ocr_check)
    samples = commands.add_parser("make-samples", help="Generate and label deterministic synthetic documents.")
    samples.add_argument("--output", type=Path, required=True)
    samples.add_argument("--seed", type=int, default=42)
    samples.add_argument("--font", type=Path)
    samples.set_defaults(handler=make_samples)
    validate = commands.add_parser("validate-data", help="Validate a dataset manifest and its files.")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--require-real-plan", action="store_true",
                          help="Require 4 development and 6 test documents with four actual photos each.")
    validate.set_defaults(handler=validate_data)
    scanning = commands.add_parser("scan", help="Scan a JPG/PNG or a directory in stable order.")
    scanning.add_argument("--input", type=Path, required=True)
    scanning.add_argument("--output", type=Path, required=True)
    scanning.add_argument("--corners", type=Path, help="A JSON file for one input, or a sidecar directory for a batch.")
    scanning.add_argument("--auto-first", action="store_true", help="Only use supplied corners after automatic detection fails.")
    scanning.add_argument("--config", type=Path)
    scanning.add_argument("--method", choices=METHODS, default="E")
    scanning.add_argument("--ocr", choices=("none", "paddle"), default="none")
    scanning.add_argument("--pdf-mode", choices=("none", "image", "searchable"), default="none")
    scanning.add_argument("--source", choices=("real", "synthetic", "unspecified"), default="unspecified")
    scanning.add_argument("--page-orientation", choices=("portrait", "landscape"), default="portrait")
    scanning.add_argument("--page-format", choices=("a4", "native"))
    scanning.add_argument("--max-long-edge", type=int)
    scanning.add_argument("--pdf-font", type=Path)
    scanning.set_defaults(handler=scan)
    pages = commands.add_parser("to-pdf", aliases=["book"], help="Convert sheets or book photos into one white-background PDF.")
    pages.add_argument("--input", type=Path, required=True, help="One JPG/PNG or a folder ordered by filename numbers.")
    pages.add_argument("--output", type=Path, required=True)
    pages.add_argument("--layout", choices=("single", "spread"), default="single")
    pages.add_argument("--gutter", type=_gutter, help="Override automatic split with one x fraction or top,bottom fractions.")
    pages.add_argument("--corners", type=Path, help="Optional corners JSON for one sheet or same-stem sidecars for a folder.")
    pages.add_argument("--page-format", choices=("native", "a4"), default="native")
    pages.add_argument("--page-orientation", choices=("auto", "portrait", "landscape"), default="auto")
    pages.add_argument("--dpi", type=int, default=300, help="Density of the embedded page image (72–400, default 300).")
    pages.add_argument("--same-page", action="store_true", help="Combine 2–4 photographs of the same complete sheet into one page.")
    pages.add_argument("--pdf-mode", choices=("image", "searchable"), default="image")
    pages.add_argument("--dewarp", choices=("none", "uvdoc"), default="none")
    pages.add_argument("--config", type=Path)
    pages.add_argument("--pdf-font", type=Path)
    pages.add_argument("--source", choices=("real", "synthetic", "unspecified"), default="unspecified")
    pages.set_defaults(handler=book)
    for command in (ocr, scanning, pages):
        command.add_argument("--reading-order", choices=("rows", "two-column"), default="rows",
                             help="Read by rows, or read left then right between headings crossing the page center.")
    pdf = commands.add_parser("verify-pdf", help="Check real text extraction, search, optional coordinates and rendering.")
    pdf.add_argument("--input", type=Path, required=True)
    pdf.add_argument("--expected-text", type=Path, required=True)
    pdf.add_argument("--ocr-json", type=Path)
    pdf.add_argument("--keyword")
    pdf.add_argument("--background", type=Path)
    pdf.add_argument("--render", type=Path)
    pdf.add_argument("--page", type=int, help="Select a page in a book for background, coordinate or render checks (1-based).")
    pdf.set_defaults(handler=verify_pdf)
    bench = commands.add_parser("benchmark", help="Evaluate A-F with shared corners or evaluate detection separately.")
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--split", choices=("dev", "test"), required=True)
    bench.add_argument("--source", choices=("real", "synthetic"), required=True)
    bench.add_argument("--mode", choices=("shared-corners", "auto"), default="shared-corners")
    bench.add_argument("--config", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    bench.set_defaults(handler=benchmark)
    freeze = commands.add_parser("freeze-config", help="Freeze a successful development run's config for test.")
    freeze.add_argument("--dev-run", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--note", default="Selected on the development split before test.")
    freeze.set_defaults(handler=freeze_config)
    summary = commands.add_parser("summarize", help="Build figures, paired cases and timing tables from a benchmark.")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.add_argument("--font", type=Path)
    summary.set_defaults(handler=summarize)
    report = commands.add_parser("build-report", help="Build a Markdown and editable Word course report.")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--auto-run-dir", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(handler=build_report)
    return root


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except (DocscanError, OSError, ImportError, ValueError, RuntimeError) as error:
        print(f"docscan: {getattr(error, 'code', type(error).__name__)}: {error}", file=sys.stderr)
        return 1
