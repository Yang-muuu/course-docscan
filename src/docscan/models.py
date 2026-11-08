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
CANDIDATE_MODELS = {"detection": MODEL_NAMES["detection"], "orientation": "PP-LCNet_x1_0_doc_ori", **DEWARP_MODELS}
MODEL_BASE = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
MODEL_FILES = {"inference.yml", "inference.json", "inference.pdiparams"}
ORIENTATION_SOURCE = "https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6"
ARCHIVE_HASHES = {
    "PP-OCRv5_mobile_det": "50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58",
    "PP-OCRv5_mobile_rec": "566b9512b34e34a9f0db54d87b51fa5a0b9ed2cf1ab7e49728cc0b8b5a64f414",
    "UVDoc": "15d79c6bcbf738b7e132eedff6ed57c7e834c73cbc140a1fff8fd8ab9c2a35b9",
}
MODEL_FILE_HASHES = {
    "PP-OCRv5_mobile_det": {
        "inference.json": "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
        "inference.pdiparams": "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
        "inference.yml": "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699",
    },
    "PP-OCRv5_mobile_rec": {
        "inference.json": "24587345250c7332d0fc6f9a44e794d078cdaeb64c302fef906f325619de2569",
        "inference.pdiparams": "2460da90875937c94db97eba74ae3d9e5d4c4c57c42f1f41531c09a26bcc771a",
        "inference.yml": "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e",
    },
    "UVDoc": {
        "inference.json": "2c2bc3e0f15e782cf8f2ad411b5033d99ca504fe88648f8054a5e925ba2336e0",
        "inference.pdiparams": "810488899520e0da843b9bd9769ba4949f1c81e357f0eceb12d4a7da459c3eca",
        "inference.yml": "be83d537b358f3ff87740e77e14a83ee9e9a7bb215c33d091b69e8bd5904fe39",
    },
    "PP-LCNet_x1_0_doc_ori": {
        "inference.json": "3580978602f309c3554508dd85d4fe09b73a7e0d80d7f9e63258f5b72c390c69",
        "inference.pdiparams": "e8d6e7c5d264507e40e58a655779059d616b20d7441ea22047d829eb3931989c",
        "inference.yml": "9e195eb729a8173588cd0e8a852c8b373aa606e79e77b4ac7d8346f5426caf26",
    },
}


def configure_paddle_cache(root: Path) -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(root / ".paddlex"))


def paddle_model_path(directory: str) -> str:
    if os.name == "nt" and not directory.isascii():
        # Paddle's Windows file reader can use an ASCII path relative to the app directory.
        path, current = Path(directory), Path.cwd().resolve()
        if path.is_relative_to(current):
            relative = str(path.relative_to(current))
            if relative.isascii():
                return relative
    return directory


def _inspect_model(folder: Path, name: str) -> dict:
    expected = MODEL_FILE_HASHES[name]
    missing, invalid, hashes = [], [], {}
    for filename in sorted(MODEL_FILES):
        path = folder / filename
        try:
            hashes[filename] = file_sha256(path)
        except FileNotFoundError:
            missing.append(filename)
        except OSError:
            invalid.append(filename)
        else:
            if hashes[filename] != expected[filename]:
                invalid.append(filename)
    return {"name": name, "directory": str(folder), "ready": not (missing or invalid), "missing": missing,
            "invalid": invalid, "file_sha256": hashes}


def model_status(root: Path, models: dict[str, str] = MODEL_NAMES) -> dict:
    return {role: _inspect_model(root / name, name) for role, name in models.items()}


def invalid_model_paths(status: dict) -> list[str]:
    return [str(Path(model["directory"]) / name) for model in status.values() for name in model["invalid"]]


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


def _download_model(name: str, stage: Path, entry: dict) -> None:
    start = perf_counter()
    if name == CANDIDATE_MODELS["orientation"]:
        target = stage / "model"
        target.mkdir()
        for filename in sorted(MODEL_FILES):
            with urlopen(f"{ORIENTATION_SOURCE}/{filename}", timeout=60) as source, (target / filename).open("wb") as output:
                shutil.copyfileobj(source, output)
        entry["download_ms"] = elapsed_ms(start)
    else:
        archive = stage / "model.tar"
        with urlopen(entry["url"], timeout=60) as source, archive.open("wb") as output:
            shutil.copyfileobj(source, output)
        entry.update(download_ms=elapsed_ms(start), archive_sha256=file_sha256(archive))
        if entry["archive_sha256"] != ARCHIVE_HASHES[name]:
            raise DocscanError("model_archive_changed", f"Unexpected SHA-256 for {name}")
        _extract_model(archive, stage / "model")


def prepare_models(root: Path, *, dewarp: bool = False, candidates: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    entries = {}
    models = CANDIDATE_MODELS if candidates else DEWARP_MODELS if dewarp else MODEL_NAMES
    for name in models.values():
        target = root / name
        url = ORIENTATION_SOURCE if name == CANDIDATE_MODELS["orientation"] else f"{MODEL_BASE}/{name}_infer.tar"
        entry = {"url": url}
        entries[name] = entry
        try:
            existing = _inspect_model(target, name)
            if existing["ready"]:
                entry["status"] = "already_available"
                continue
            if existing["invalid"]:
                paths = ", ".join(invalid_model_paths({name: existing}))
                raise DocscanError("model_file_changed", f"Cache checksum or read error: {paths}. Move {target} aside and prepare again.")
            if target.exists():
                raise DocscanError("incomplete_model_dir", f"Move the incomplete model directory aside: {target}")
            with TemporaryDirectory(dir=root) as staging:
                stage = Path(staging)
                _download_model(name, stage, entry)
                if not _inspect_model(stage / "model", name)["ready"]:
                    raise DocscanError("model_file_changed", f"Extracted model files failed checksum verification: {name}")
                (stage / "model").rename(target)
                entry["status"] = "downloaded"
        except (OSError, tarfile.TarError, DocscanError) as error:
            entry.update(status="error", **error_record(error))
    status = "error" if any(value["status"] == "error" for value in entries.values()) else "ok"
    result = {"status": status, "models": entries, "finished_at_utc": utc_now(), "elapsed_ms": elapsed_ms(start)}
    if status == "ok":
        result["model_file_sha256"] = model_hashes(root, models)
    record_name = "candidate_preparation.json" if candidates else "dewarp_preparation.json" if dewarp else "model_preparation.json"
    record = root / record_name
    if not record.exists() or any(value["status"] != "already_available" for value in entries.values()):
        write_json(record, result)
    return result
