from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

SEED = 42
METHODS = tuple("ABCDEF")
CONDITIONS = ("front_even", "tilt_even", "front_shadow", "tilt_shadow")
DEPENDENCIES = (
    "course-docscan", "numpy", "opencv-contrib-python", "Pillow", "PyYAML",
    "PyMuPDF", "paddlepaddle", "paddleocr", "paddlex", "matplotlib", "python-docx",
)


class DocscanError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def error_record(error: Exception) -> dict:
    return {"error_code": getattr(error, "code", "unexpected_error"), "error": str(error)}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 3)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, timeout=5)
            return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout) if not status.returncode else None}
        return {"commit": None, "dirty": None, "reason": "Git HEAD unavailable in working directory"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "reason": "Git command unavailable"}


def hardware() -> dict:
    return {"platform": platform.platform(), "architecture": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count()}


def new_run(output: Path) -> tuple[Path, str]:
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    try:
        output.mkdir(parents=True)
    except FileExistsError:
        output = output / run_id
        output.mkdir()
    return output, run_id


def run_metadata(run_id: str, config: dict, source: str) -> dict:
    return {
        "run_id": run_id, "started_at_utc": utc_now(), "status": "in_progress", "source": source,
        "effective_config": config, "config_sha256": json_sha256(config), "git": git_state(),
        "python": sys.version.split()[0], "dependencies": dependency_versions(), "hardware": hardware(),
        "random_seed": SEED, "elapsed_ms": {}, "outputs": {},
    }


def finish_run(path: Path, metadata: dict) -> None:
    metadata["finished_at_utc"] = utc_now()
    write_json(path / "run.json", metadata)

