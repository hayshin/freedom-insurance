from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import polars as pl


def read_input(path: str | Path, nrows: int | None) -> pl.DataFrame:
    return pl.read_csv(path, n_rows=nrows, infer_schema_length=None, encoding="utf8-lossy")


def frame_len(frame: pl.DataFrame) -> int:
    return len(frame)


def polars_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))


def save_pickle(path: Path, obj) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)
