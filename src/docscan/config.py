from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import yaml

from .common import DocscanError

DEFAULTS = {
    "geometry": {"page_format": "a4", "max_long_edge": 1800, "detect_max_side": 960,
                 "min_area_ratio": 0.18, "max_area_ratio": 0.96},
    "enhance": {"background_scale": 0.045, "background_min": 15, "background_max": 121,
                "background_floor": 16.0, "target_white": 245.0, "adaptive_block": 41,
                "adaptive_c": 12.0},
    "ocr": {"model_dir": ".models", "device": "cpu", "cpu_threads": 4,
            "max_input_side": 1600, "score_thresh": 0.0},
    "pdf": {"dpi": 150.0, "font": None},
    "evaluation": {"auto_iou_min": 0.92, "auto_corner_error_max": 0.03},
}


def load_config(path: Path | None = None) -> tuple[dict, dict | None]:
    try:
        overlay = yaml.safe_load(path.read_text(encoding="utf-8-sig")) if path else {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise DocscanError("invalid_config", f"Cannot read {path}: {error}") from error
    if not isinstance(overlay, dict):
        raise DocscanError("invalid_config", "Expected a YAML mapping")
    freeze = overlay.pop("freeze", None)
    if freeze is not None and not isinstance(freeze, dict):
        raise DocscanError("invalid_config", "freeze must be a mapping")
    config = deepcopy(DEFAULTS)
    for section, values in overlay.items():
        if section not in config or not isinstance(values, dict):
            raise DocscanError("invalid_config", f"Unknown or invalid section {section!r}")
        for key, value in values.items():
            if key not in config[section]:
                raise DocscanError("invalid_config", f"Unknown option {section}.{key}")
            default = config[section][key]
            if isinstance(default, float) and type(value) in (int, float) and math.isfinite(value):
                value = float(value)
            elif default is None and (value is None or isinstance(value, str) and value.strip()):
                pass
            elif default is None or type(value) is not type(default) or isinstance(value, float):
                raise DocscanError("invalid_config", f"Invalid value for {section}.{key}: {value!r}")
            config[section][key] = value
    validate_config(config)
    return config, freeze


def validate_config(config: dict) -> None:
    geo, enh, ocr, pdf, ev = (config[key] for key in DEFAULTS)
    valid = {
        "geometry.page_format": geo["page_format"] in {"a4", "native"},
        "geometry.max_long_edge": 32 <= geo["max_long_edge"] <= 12000,
        "geometry.detect_max_side": 128 <= geo["detect_max_side"] <= 4096,
        "geometry.area_ratios": 0 < geo["min_area_ratio"] < geo["max_area_ratio"] <= 1,
        "enhance.background_scale": 0 < enh["background_scale"] <= 1,
        "enhance.background_min/max": 3 <= enh["background_min"] <= enh["background_max"] <= 501
        and enh["background_min"] % 2 == enh["background_max"] % 2 == 1,
        "enhance.background_floor": 0 < enh["background_floor"] <= 255,
        "enhance.target_white": 0 < enh["target_white"] <= 255,
        "enhance.adaptive_block": 3 <= enh["adaptive_block"] <= 501 and enh["adaptive_block"] % 2 == 1,
        "enhance.adaptive_c": 0 < enh["adaptive_c"] <= 255,
        "ocr.model_dir": bool(ocr["model_dir"].strip()),
        "ocr.device": ocr["device"] == "cpu",
        "ocr.cpu_threads": 1 <= ocr["cpu_threads"] <= 64,
        "ocr.max_input_side": 128 <= ocr["max_input_side"] <= 4096,
        "ocr.score_thresh": 0 <= ocr["score_thresh"] <= 1,
        "pdf.dpi": 30 <= pdf["dpi"] <= 1200,
        "evaluation.auto_iou_min": 0 < ev["auto_iou_min"] <= 1,
        "evaluation.auto_corner_error_max": 0 < ev["auto_corner_error_max"] <= 1,
    }
    for option, is_valid in valid.items():
        if not is_valid:
            raise DocscanError("invalid_config", f"Out-of-range or unsupported option: {option}")

