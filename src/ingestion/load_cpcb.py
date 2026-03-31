from pathlib import Path
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS: List[str] = [
    "Timestamp",
    "station_name",
    "PM2.5 (µg/m³)",
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "WD (deg)"
]


COLUMN_RENAME_MAP = {
    "Timestamp": "timestamp",
    "station_name": "station",
    "PM2.5 (µg/m³)": "pm25",
    "AT (°C)": "temperature",
    "RH (%)": "humidity",
    "WS (m/s)": "wind_speed",
    "WD (deg)": "wind_direction",
}


def load_cpcb(raw_path: Path) -> pd.DataFrame:
    """Read raw CPCB/CAAQMS CSV and standardize column names and datetime."""
    logger.info(f"Loading raw CPCB data from {raw_path}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_csv(raw_path)

    _validate_required_columns(df)

    df = df[list(REQUIRED_COLUMNS)].copy()

    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=False,
        infer_datetime_format=True
    )

    _validate_timestamps(df)

    logger.info(
        "Successfully loaded raw CPCB data "
        f"({len(df)} rows, {df['station'].nunique()} stations)"
    )

    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError when required columns are missing."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")


def _validate_timestamps(df: pd.DataFrame) -> None:
    """Raise ValueError when timestamp conversion fails."""
    if df["timestamp"].isna().any():
        n_bad = df["timestamp"].isna().sum()
        raise ValueError(
            f"{n_bad} rows have invalid timestamps after parsing"
        )


def preprocess_cpcb(df: pd.DataFrame, max_interpolate_gap: int = 3) -> pd.DataFrame:
    """Apply standard preprocessing like interpolation and feature generation."""
    if df.empty:
        raise ValueError("DataFrame is empty; nothing to preprocess")

    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # interpolate the most common columns from notebook and keep consistent values
    interpolate_columns = ["pm25", "temperature", "humidity", "wind_speed", "wind_direction"]

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

    # time features for modeling
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    logger.info("Preprocessing completed: interpolation + feature generation")

    return df


def save_cleaned_data(df: pd.DataFrame, clean_path: Path) -> Path:
    """Write cleaned DataFrame to csv path."""
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    logger.info(f"Cleaned data saved to {clean_path}")
    return clean_path


def run_etl(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """End-to-end ingestion + preprocessing + save pipeline."""
    raw_df = load_cpcb(raw_path)
    clean_df = preprocess_cpcb(raw_df)
    save_cleaned_data(clean_df, clean_path)
    return clean_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    base = Path(__file__).resolve().parents[2]
    raw_csv = base / "data" / "raw" / "bangalore_caaqms" / "bangalore_caaqms_2025_combined.csv"
    clean_csv = base / "data" / "clean" / "bangalore_clean.csv"

    run_etl(raw_csv, clean_csv)


if __name__ == "__main__":
    main()
