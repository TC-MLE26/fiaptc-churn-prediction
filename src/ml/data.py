"""Dataset download and raw file loading."""

from __future__ import annotations
import os
import shutil
from pathlib import Path
import pandas as pd

KAGGLE_DATASET = "yeanzc/telco-customer-churn-ibm-dataset"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EXCEL_SUFFIXES = (".xlsx", ".xls")
SUPPORTED_SUFFIXES = EXCEL_SUFFIXES + (".csv",)


def download_dataset(dataset: str = KAGGLE_DATASET) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(dataset))


def find_dataset_file(directory: str | Path) -> Path:
    candidates = [
        Path(root) / name
        for root, _, files in os.walk(directory)
        for name in files
        if name.lower().endswith(SUPPORTED_SUFFIXES)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx, .xls or .csv file found in '{directory}'"
        )

    return max(
        candidates,
        key=lambda path: (path.suffix.lower() in EXCEL_SUFFIXES, path.stat().st_size),
    )


def read_dataset_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() in EXCEL_SUFFIXES:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file extension: '{path.suffix}'")


def ensure_dataset(directory: str | Path = DATA_DIR) -> Path:
    directory = Path(directory)

    try:
        return find_dataset_file(directory)
    except FileNotFoundError:
        pass

    downloaded = find_dataset_file(download_dataset())
    directory.mkdir(parents=True, exist_ok=True)
    local_copy = directory / downloaded.name
    shutil.copy2(downloaded, local_copy)

    return local_copy


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    return read_dataset_file(path if path is not None else ensure_dataset())
