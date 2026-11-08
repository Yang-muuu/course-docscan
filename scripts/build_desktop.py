"""Build the Windows x64 program with an application-local CPython runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
MSVC_RUNTIME = ("msvcp140.dll", "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
sys.path.insert(0, str(ROOT / "src"))

from docscan.models import CANDIDATE_MODELS, MODEL_FILES, MODEL_NAMES, model_status


def copy_locked_packages(destination: Path) -> None:
    for line in (ROOT / "requirements.win-py313.lock").read_text(encoding="utf-8").splitlines():
        name, version = line.split("==")
        distribution = importlib.metadata.distribution(name)
        if distribution.version != version:
            raise RuntimeError(f"{name} must be {version}, found {distribution.version}")
        for item in distribution.files or ():
            # Wheels also record generated scripts outside site-packages.
            if item.is_absolute() or ".." in item.parts or item.suffix in {".pyc", ".pth"}:
                continue
            source = distribution.locate_file(item)
            if not source.is_file():
                raise RuntimeError(f"Installed package is incomplete: {name}/{item}")
            target = destination / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def build(output: Path, model_directory: Path) -> None:
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8:
        raise RuntimeError("Build with the project's Windows x64 Python 3.13 environment.")
    if output.exists():
        raise FileExistsError(output)
    models = {**MODEL_NAMES, **CANDIDATE_MODELS}
    states = model_status(model_directory, models)
    if any(not state["ready"] for state in states.values()):
        raise RuntimeError("Prepare and verify the required local models before building.")

    base = Path(sys.base_prefix)
    windows = Path(os.environ["WINDIR"])
    csc = windows / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    redist = windows / "System32"
    if not all((redist / name).is_file() for name in MSVC_RUNTIME):
        raise RuntimeError("Install the Microsoft Visual C++ Redistributable (x64) on the build machine.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="docscan-build-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "DocScan"
        runtime = bundle / "runtime"
        runtime.mkdir(parents=True)
        for filename in ("python.exe", "pythonw.exe", "python3.dll", "python313.dll", "LICENSE.txt"):
            shutil.copy2(base / filename, runtime / filename)
        for filename in MSVC_RUNTIME:
            shutil.copy2(redist / filename, runtime / filename)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdb", "site-packages", "ensurepip", "idlelib", "test")
        shutil.copytree(base / "Lib", runtime / "Lib", ignore=ignore)
        shutil.copytree(base / "DLLs", runtime / "DLLs", ignore=shutil.ignore_patterns("_test*", "*.pdb"))
        for name in ("tcl8.6", "tk8.6", "tcl8"):
            shutil.copytree(base / "tcl" / name, runtime / "tcl" / name)
        (runtime / "python313._pth").write_text(
            ".\nLib\nDLLs\nLib/site-packages\n../app\n", encoding="utf-8",
        )
        copy_locked_packages(runtime / "Lib/site-packages")
        shutil.copytree(ROOT / "src/docscan", bundle / "app/docscan", ignore=ignore)
        shutil.copytree(ROOT / "examples/real", bundle / "examples/real")
        shutil.copy2(ROOT / "examples/real/README.md", bundle / "README.md")
        shutil.copy2(ROOT / "THIRD_PARTY.md", bundle / "THIRD_PARTY.md")
        for name in models.values():
            destination = bundle / "models" / name
            destination.mkdir(parents=True)
            for filename in MODEL_FILES:
                shutil.copy2(model_directory / name / filename, destination / filename)
        subprocess.run(
            [str(csc), "/nologo", "/target:winexe", "/platform:x64", "/reference:System.Windows.Forms.dll",
             f"/out:{bundle / 'DocScan.exe'}", str(ROOT / "scripts/DocScanLauncher.cs")],
            check=True,
        )
        bundle.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/DocScan")
    parser.add_argument("--models", type=Path, default=ROOT / ".models")
    args = parser.parse_args()
    build(args.output.resolve(), args.models.resolve())
    print(args.output.resolve() / "DocScan.exe")


if __name__ == "__main__":
    main()
