from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.request import urlopen

from .common import DocscanError, elapsed_ms, error_record, file_sha256, utc_now, write_json

MODEL_NAMES = {"detection": "PP-OCRv5_mobile_det", "recognition": "PP-OCRv5_mobile_rec"}
DEWARP_MODELS = {"dewarp": "UVDoc"}
MODEL_BASE = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
MODEL_FILES = {"inference.yml", "inference.json", "inference.pdiparams"}
ARCHIVE_HASHES = {
    "PP-OCRv5_mobile_det": "50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58",
    "PP-OCRv5_mobile_rec": "566b9512b34e34a9f0db54d87b51fa5a0b9ed2cf1ab7e49728cc0b8b5a64f414",
    "UVDoc": "15d79c6bcbf738b7e132eedff6ed57c7e834c73cbc140a1fff8fd8ab9c2a35b9",
}


def configure_paddle_cache(root: Path) -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(root / ".paddlex"))


def model_status(root: Path, models: dict[str, str] = MODEL_NAMES) -> dict:
    result = {}
    for role, name in models.items():
        folder = root / name
        missing = sorted(item for item in MODEL_FILES if not (folder / item).is_file())
        result[role] = {"name": name, "directory": str(folder), "ready": not missing, "missing": missing}
    return result


def model_hashes(root: Path, models: dict[str, str] = MODEL_NAMES) -> dict:
    return {role: {name: file_sha256(root / model / name) for name in sorted(MODEL_FILES)}
            for role, model in models.items()}


def _extract_model(tar_path: Path, target: Path) -> None:
    target.mkdir()
    with tarfile.open(tar_path, "r:*") as archive:
        for item in archive:
            basename = PurePosixPath(item.name).name
            if basename in MODEL_FILES and item.isfile():
                with archive.extractfile(item) as source, (target / basename).open("xb") as output:
                    shutil.copyfileobj(source, output)
    if not all((target / item).is_file() for item in MODEL_FILES):
        raise DocscanError("invalid_model_archive", f"Model files missing from {tar_path.name}")


def prepare_models(root: Path, *, dewarp: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    entries = {}
    models = DEWARP_MODELS if dewarp else MODEL_NAMES
    for name in models.values():
        target = root / name
        url = f"{MODEL_BASE}/{name}_infer.tar"
        entry = {"url": url}
        entries[name] = entry
        try:
            if all((target / item).is_file() for item in MODEL_FILES):
                entry["status"] = "already_available"
                continue
            if target.exists():
                raise DocscanError("incomplete_model_dir", f"Move the incomplete model directory aside: {target}")
            with TemporaryDirectory(dir=root) as staging:
                stage = Path(staging)
                tar_path = stage / "model.tar"
                tick = perf_counter()
                with urlopen(url, timeout=60) as response, tar_path.open("wb") as file:
                    shutil.copyfileobj(response, file)
                entry.update(download_ms=elapsed_ms(tick), archive_sha256=file_sha256(tar_path))
                if entry["archive_sha256"] != ARCHIVE_HASHES[name]:
                    raise DocscanError("model_archive_changed", f"Unexpected SHA-256 for {name}")
                _extract_model(tar_path, stage / "model")
                (stage / "model").rename(target)
                entry["status"] = "downloaded"
        except (OSError, tarfile.TarError, DocscanError) as error:
            entry.update(status="error", **error_record(error))
    status = "error" if any(value["status"] == "error" for value in entries.values()) else "ok"
    result = {"status": status, "models": entries, "finished_at_utc": utc_now(), "elapsed_ms": elapsed_ms(start)}
    if status == "ok":
        result["model_file_sha256"] = model_hashes(root, models)
    record = root / ("dewarp_preparation.json" if dewarp else "model_preparation.json")
    if not record.exists() or any(value["status"] != "already_available" for value in entries.values()):
        write_json(record, result)
    return result
