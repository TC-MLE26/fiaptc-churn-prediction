import pandas as pd
import pytest

from src.ml import data
from src.ml.data import (
    ensure_dataset,
    find_dataset_file,
    load_raw,
    read_dataset_file,
)


def _write_csv(path, rows=1):
    pd.DataFrame({"a": range(rows)}).to_csv(path, index=False)
    return path


def test_find_dataset_file_prefers_excel(tmp_path):
    _write_csv(tmp_path / "big.csv", rows=500)
    pd.DataFrame({"a": [1]}).to_excel(tmp_path / "small.xlsx", index=False)

    assert find_dataset_file(tmp_path).name == "small.xlsx"


def test_find_dataset_file_picks_the_largest_among_equals(tmp_path):
    _write_csv(tmp_path / "small.csv", rows=1)
    _write_csv(tmp_path / "large.csv", rows=500)

    assert find_dataset_file(tmp_path).name == "large.csv"


def test_find_dataset_file_searches_subdirectories(tmp_path):
    nested = tmp_path / "raw"
    nested.mkdir()
    _write_csv(nested / "dataset.csv")

    assert find_dataset_file(tmp_path).name == "dataset.csv"


def test_find_dataset_file_without_candidates_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("not a dataset", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        find_dataset_file(tmp_path)


def test_read_dataset_file_reads_csv(tmp_path):
    path = _write_csv(tmp_path / "dataset.csv", rows=3)

    assert len(read_dataset_file(path)) == 3


def test_read_dataset_file_reads_excel(tmp_path):
    path = tmp_path / "dataset.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(path, index=False)

    assert len(read_dataset_file(path)) == 2


def test_read_dataset_file_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_dataset_file(path)


def test_ensure_dataset_uses_local_file_without_downloading(tmp_path, monkeypatch):
    _write_csv(tmp_path / "dataset.csv")

    def fail_download(*args, **kwargs):
        raise AssertionError("download_dataset should not be called")

    monkeypatch.setattr(data, "download_dataset", fail_download)

    assert ensure_dataset(tmp_path).name == "dataset.csv"


def test_ensure_dataset_downloads_and_caches_locally(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _write_csv(remote / "downloaded.csv", rows=2)

    local = tmp_path / "data"
    monkeypatch.setattr(data, "download_dataset", lambda *args, **kwargs: remote)

    result = ensure_dataset(local)

    assert result == local / "downloaded.csv"
    assert result.is_file()


def test_load_raw_reads_the_given_path(tmp_path):
    path = _write_csv(tmp_path / "dataset.csv", rows=4)

    assert len(load_raw(path)) == 4
