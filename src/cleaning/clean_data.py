import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, max_interpolate_gap: int = 3) -> pd.DataFrame:
    """Apply data cleaning: sorting, interpolation, and row filtering."""
    if df.empty:
        raise ValueError("DataFrame is empty; nothing to clean")

    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # interpolate the most common columns from notebook and keep consistent values
    interpolate_columns: List[str] = ["pm25", "temperature", "humidity", "wind_speed", "wind_direction"]

    # We cannot use method='time' rolling interpolation on a grouped Series unless index is datetime.
    # Use linear interpolation to avoid DatetimeIndex requirement, with a small limit to avoid extrapolation.
    # Use transform() to keep the same index alignment as the parent DataFrame.
    for col in interpolate_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column missing from input: {col}")

        df[col] = df.groupby("station")[col].transform(
            lambda x: x.interpolate(method="linear", limit=max_interpolate_gap)
        )

    df = df.dropna(subset=["pm25", "temperature", "humidity", "wind_speed", "wind_direction", "timestamp"])

    logger.info("Data cleaning completed: sorting, interpolation, and filtering")

    return df
