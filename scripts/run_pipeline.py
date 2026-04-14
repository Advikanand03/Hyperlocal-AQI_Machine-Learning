import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cleaning.clean_data import clean_data
from src.features.temporal import add_temporal_features
from src.ingestion.load_cpcb import load_cpcb


TOP_N_STATIONS = 6


def get_project_paths(base_path: Path) -> dict[str, Path]:
    """Return all file paths used by the processing pipeline."""
    return {
        "raw": base_path / "data" / "raw" / "bangalore_caaqms" / "bangalore_caaqms_2025_combined.csv",
        "metadata": base_path / "data" / "metadata" / "bangalore_stations.csv",
        "processed": base_path / "data" / "processed" / "bangalore_processed.csv",
    }


def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """Load the full raw Bengaluru dataset containing all stations."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    return pd.read_csv(raw_path)


def select_top_stations(raw_df: pd.DataFrame, top_n: int = TOP_N_STATIONS) -> pd.DataFrame:
    """Rank stations by row count and flag the stations selected for modeling."""
    if "station_name" not in raw_df.columns:
        raise ValueError("Raw dataset must contain 'station_name' column")

    station_counts = (
        raw_df.groupby("station_name")
        .size()
        .sort_values(ascending=False)
        .rename("row_count")
        .reset_index()
    )
    station_counts["rank"] = range(1, len(station_counts) + 1)
    station_counts["selected"] = station_counts["rank"] <= top_n
    return station_counts


def attach_station_coordinates(
    processed_df: pd.DataFrame,
    metadata_path: Path,
    selected_stations: list[str],
) -> pd.DataFrame:
    """Join latitude and longitude metadata for the selected stations."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Station metadata not found: {metadata_path}")

    stations_df = pd.read_csv(metadata_path)
    required_columns = {"station", "latitude", "longitude"}
    missing_columns = required_columns - set(stations_df.columns)
    if missing_columns:
        raise ValueError(f"Station metadata missing required columns: {missing_columns}")

    missing_metadata = sorted(set(selected_stations) - set(stations_df["station"]))
    if missing_metadata:
        raise ValueError(
            "Selected stations missing from station metadata: "
            + ", ".join(missing_metadata)
        )

    stations_df = stations_df[["station", "latitude", "longitude"]].drop_duplicates()
    merged_df = processed_df.merge(stations_df, on="station", how="left")

    if merged_df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing latitude/longitude values after metadata merge")

    return merged_df


def build_processed_dataset(base_path: Path, top_n: int = TOP_N_STATIONS):
    """Select top stations, preprocess them, attach coordinates, and save output."""
    paths = get_project_paths(base_path)
    raw_df = load_raw_dataset(paths["raw"])
    station_counts = select_top_stations(raw_df, top_n=top_n)
    selected_stations = station_counts.loc[station_counts["selected"], "station_name"].tolist()

    logging.info("=" * 70)
    logging.info("STATION SELECTION")
    logging.info("=" * 70)
    logging.info(
        "Selecting top %s stations by raw row count from %s total stations:",
        top_n,
        raw_df["station_name"].nunique(),
    )
    for row in station_counts.itertuples(index=False):
        marker = "SELECTED" if row.selected else "excluded"
        logging.info("  %2d. %-35s %7s rows  %s", row.rank, row.station_name, f"{row.row_count:,}", marker)
    logging.info("")

    standardized_df = load_cpcb(paths["raw"])
    standardized_df = standardized_df[standardized_df["station"].isin(selected_stations)].copy()

    processed_df = clean_data(standardized_df)
    processed_df = add_temporal_features(processed_df)
    processed_df = attach_station_coordinates(processed_df, paths["metadata"], selected_stations)

    paths["processed"].parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(paths["processed"], index=False)

    logging.info("Processed dataset created: %s", processed_df.shape)
    logging.info("Processed dataset saved to %s", paths["processed"])
    logging.info("Selected stations: %s", ", ".join(selected_stations))

    return raw_df, processed_df, station_counts


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = Path(__file__).resolve().parents[1]
    build_processed_dataset(base)


if __name__ == "__main__":
    main()
