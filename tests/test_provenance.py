import shutil
import subprocess

import pytest

from docscan import common


def _init_repository(path, filename):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / filename).write_text("Repository fixture\n", encoding="utf-8")
    git = ["git", "-C", str(path)]
    subprocess.run([*git, "add", filename], check=True)
    subprocess.run([*git, "-c", "user.name=Test Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Fixture"], check=True)
    return subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()


def test_git_provenance_comes_from_loaded_source(tmp_path, monkeypatch):
    if not shutil.which("git"):
        pytest.skip("Git is required to check source provenance")
    source, caller = tmp_path / "source", tmp_path / "caller"
    source_commit = _init_repository(source, "common.py")
    assert _init_repository(caller, "unrelated.txt") != source_commit
    monkeypatch.chdir(caller)
    monkeypatch.setattr(common, "__file__", str(source / "common.py"))
    assert common.git_state() == {"commit": source_commit, "dirty": False}
    untracked = caller / "installed.py"
    untracked.write_text("Untracked install fixture\n", encoding="utf-8")
    monkeypatch.setattr(common, "__file__", str(untracked))
    assert common.git_state()["commit"] is None
