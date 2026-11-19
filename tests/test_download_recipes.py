"""download_recipes.sh must never leave a file that fails its checksum in data/.

Before, a bad copy or an interrupted download stayed in place, the next run skipped it because
the file existed, and every later run failed the checksum until someone deleted it by hand.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "download_recipes.sh"
REAL = REPO / "data"
OR_NAME = "20170107-061401-recipeitems.json.gz"
GISMO = ["vocab_ingrs.pkl", "train_comments_subs.pkl", "val_comments_subs.pkl", "test_comments_subs.pkl"]
needs_data = pytest.mark.skipif(
    not all((REAL / "recipe1msubs" / n).exists() for n in GISMO)
    or not (REAL / "openrecipes" / OR_NAME).exists(),
    reason="run ./download_recipes.sh",
)


def run(data, **env):
    return subprocess.run(["bash", str(SCRIPT), str(data)], capture_output=True, text=True,
                          env={**os.environ, "LC_ALL": "C", **env})


def test_bad_local_copy_is_not_kept(tmp_path):
    bad = tmp_path / "wrong.json.gz"
    bad.write_bytes(b"not the snapshot")
    result = run(tmp_path / "data", OPENRECIPES_GZ=str(bad))
    assert result.returncode != 0
    assert "checksum" in result.stderr
    assert list((tmp_path / "data" / "openrecipes").iterdir()) == []


def test_corrupt_file_from_an_earlier_run_is_replaced(tmp_path):
    data = tmp_path / "data"
    (data / "openrecipes").mkdir(parents=True)
    (data / "openrecipes" / OR_NAME).write_bytes(b"half a download")
    bad = tmp_path / "wrong.json.gz"
    bad.write_bytes(b"still not the snapshot")
    result = run(data, OPENRECIPES_GZ=str(bad))
    assert result.returncode != 0
    assert list((data / "openrecipes").iterdir()) == []  # neither the old nor the new bad file


@needs_data
def test_valid_files_are_kept_and_not_fetched_again(tmp_path):
    data = tmp_path / "data"
    (data / "openrecipes").mkdir(parents=True)
    (data / "recipe1msubs").mkdir()
    shutil.copy(REAL / "openrecipes" / OR_NAME, data / "openrecipes" / OR_NAME)
    for name in GISMO:
        shutil.copy(REAL / "recipe1msubs" / name, data / "recipe1msubs" / name)
    before = {p: p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()}
    # A path that does not exist: the script must not need it when the files are already valid.
    result = run(data, OPENRECIPES_GZ=str(tmp_path / "missing.json.gz"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(": OK") == 5
    assert {p: p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()} == before
