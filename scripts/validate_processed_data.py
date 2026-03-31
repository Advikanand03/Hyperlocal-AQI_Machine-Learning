import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def setup_logging():
    """Configure logging for clear validation output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_datasets(base_path: Path):
    """Load both raw and processed datasets."""
    raw_path = base_path / "data" / "raw" / "bangalore_caaqms" / "bangalore_caaqms_2025_combined.csv"
    processed_path = base_path / "data" / "processed" / "bangalore_processed.csv"

    logger.info("Loading datasets...")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    raw_df = pd.read_csv(raw_path)
    processed_df = pd.read_csv(processed_path)

    # Ensure timestamp is parsed
    processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"])

    logger.info(f"✓ Raw dataset loaded: {raw_df.shape}")
    logger.info(f"✓ Processed dataset loaded: {processed_df.shape}\n")

    return raw_df, processed_df


def validate_row_count(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    """Compare row counts and report data reduction."""
    logger.info("=" * 70)
    logger.info("ROW COUNT COMPARISON")
    logger.info("=" * 70)

    raw_count = len(raw_df)
    processed_count = len(processed_df)
    reduction = raw_count - processed_count
    reduction_pct = (reduction / raw_count) * 100

    logger.info(f"Raw dataset:       {raw_count:,} rows")
    logger.info(f"Processed dataset: {processed_count:,} rows")
    logger.info(f"Rows removed:      {reduction:,} ({reduction_pct:.1f}%)")
    logger.info(f"Rows retained:     {processed_count:,} ({100 - reduction_pct:.1f}%)")
    logger.info("")


def validate_station_coverage(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    """Compare station coverage."""
    logger.info("=" * 70)
    logger.info("STATION COVERAGE")
    logger.info("=" * 70)

    raw_stations = raw_df["station_name"].nunique()
    processed_stations = processed_df["station"].nunique()

    logger.info(f"Raw dataset stations:       {raw_stations}")
    logger.info(f"Processed dataset stations: {processed_stations}")
    logger.info("")

    # Per-station row counts
    logger.info("Per-station row counts (processed):")
    station_counts = processed_df.groupby("station").size().sort_values(ascending=False)
    for station, count in station_counts.items():
        logger.info(f"  {station:30s} {count:6,} rows")
    logger.info("")


def validate_date_range(processed_df: pd.DataFrame):
    """Report temporal coverage of processed data."""
    logger.info("=" * 70)
    logger.info("TEMPORAL COVERAGE")
    logger.info("=" * 70)

    min_date = processed_df["timestamp"].min()
    max_date = processed_df["timestamp"].max()
    date_range = max_date - min_date

    logger.info(f"Start date: {min_date}")
    logger.info(f"End date:   {max_date}")
    logger.info(f"Duration:   {date_range.days} days ({date_range.days / 30:.1f} months)")
    logger.info("")


def validate_missing_values(processed_df: pd.DataFrame):
    """Report missing values per column."""
    logger.info("=" * 70)
    logger.info("MISSING VALUES")
    logger.info("=" * 70)

    missing = processed_df.isnull().sum()
    missing_pct = (missing / len(processed_df)) * 100

    if missing.sum() == 0:
        logger.info("✓ No missing values detected.")
        logger.info("")
    else:
        logger.info("Missing values per column:")
        for col in processed_df.columns:
            if missing[col] > 0:
                logger.info(f"  {col:20s} {missing[col]:6,} rows ({missing_pct[col]:.2f}%)")
        logger.info("")


def validate_statistics(processed_df: pd.DataFrame):
    """Report descriptive statistics for key variables."""
    logger.info("=" * 70)
    logger.info("DESCRIPTIVE STATISTICS (Key Variables)")
    logger.info("=" * 70)

    key_vars = ["pm25", "temperature", "humidity", "wind_speed", "wind_direction"]

    for var in key_vars:
        if var not in processed_df.columns:
            continue

        data = processed_df[var].dropna()
        logger.info(f"\n{var.upper()}:")
        logger.info(f"  Count:      {len(data):,}")
        logger.info(f"  Mean:       {data.mean():.2f}")
        logger.info(f"  Std Dev:    {data.std():.2f}")
        logger.info(f"  Min:        {data.min():.2f}")
        logger.info(f"  25%:        {data.quantile(0.25):.2f}")
        logger.info(f"  Median:     {data.median():.2f}")
        logger.info(f"  75%:        {data.quantile(0.75):.2f}")
        logger.info(f"  Max:        {data.max():.2f}")

    logger.info("")


def validate_hourly_coverage(processed_df: pd.DataFrame):
    """Report hourly coverage summary."""
    logger.info("=" * 70)
    logger.info("HOURLY COVERAGE")
    logger.info("=" * 70)

    if "hour" not in processed_df.columns:
        logger.info("Hour column not found. Skipping hourly coverage.")
        logger.info("")
        return

    hourly_counts = processed_df.groupby("hour").size()

    logger.info("Rows per hour (0-23):")
    for hour in range(24):
        count = hourly_counts.get(hour, 0)
        bar = "█" * (count // 100) if count > 0 else ""
        logger.info(f"  Hour {hour:2d}: {count:6,} rows {bar}")

    logger.info("")


def validate_monthly_coverage(processed_df: pd.DataFrame):
    """Report monthly coverage summary."""
    logger.info("=" * 70)
    logger.info("MONTHLY COVERAGE")
    logger.info("=" * 70)

    if "month" not in processed_df.columns:
        logger.info("Month column not found. Skipping monthly coverage.")
        logger.info("")
        return

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    monthly_counts = processed_df.groupby("month").size()

    logger.info("Rows per month:")
    for month in range(1, 13):
        count = monthly_counts.get(month, 0)
        month_name = month_names.get(month, f"Month {month}")
        bar = "█" * (count // 500) if count > 0 else ""
        logger.info(f"  {month_name:3s}: {count:6,} rows {bar}")

    logger.info("")


def validate_completeness_per_station(processed_df: pd.DataFrame):
    """Check data completeness per station and hour."""
    logger.info("=" * 70)
    logger.info("COMPLETENESS ANALYSIS (Station × Hour Grid)")
    logger.info("=" * 70)

    if "hour" not in processed_df.columns:
        logger.info("Hour column not found. Skipping completeness analysis.")
        logger.info("")
        return

    # Create a pivot table: stations × hours
    completeness = processed_df.groupby(["station", "hour"]).size().unstack(fill_value=0)

    logger.info(f"Station coverage by hour (24 hours per station expected):")
    completeness_pct = (completeness > 0).sum(axis=1) / 24 * 100

    for station in sorted(completeness_pct.index):
        pct = completeness_pct[station]
        logger.info(f"  {station:30s} {pct:5.1f}% hourly coverage")

    logger.info("")


def main():
    base = Path(__file__).resolve().parents[1]

    try:
        raw_df, processed_df = load_datasets(base)

        validate_row_count(raw_df, processed_df)
        validate_station_coverage(raw_df, processed_df)
        validate_date_range(processed_df)
        validate_missing_values(processed_df)
        validate_statistics(processed_df)
        validate_hourly_coverage(processed_df)
        validate_monthly_coverage(processed_df)
        validate_completeness_per_station(processed_df)

        logger.info("=" * 70)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
