"""Data loading / writing utilities"""
from pathlib import Path
from typing import List, Text

import pandas as pd


def load_parquet(path: Path, columns: List[Text] = None, engine: Text = 'pyarrow', **kwargs) -> pd.DataFrame:
    """Load and retrieve parquet dataset

    :param path: path to parquet data
    :param columns: columns to load, defaults to None
    :param engine: parquet engine to load: 'pyarrow' or 'fastparquet'. PyArrow is more powerful, defaults to 'pyarrow'
    :param **kwargs: other keyword arguments to pass to pd.read_parquet
    :return: Dataset Dataframe
    """
    return pd.read_parquet(path=path, columns=columns, engine=engine, **kwargs)


def write_parquet(path: Path, data_df: pd.DataFrame, engine: Text = 'pyarrow'):
    """Write dataset to parquet

    :param path: Path to write to
    :param data_df: Dataset dataframe
    :param engine: parquet engine to load: 'pyarrow' or 'fastparquet'. PyArrow is more powerful, defaults to 'pyarrow'
    """
    data_df.to_parquet(path=path, engine=engine)
