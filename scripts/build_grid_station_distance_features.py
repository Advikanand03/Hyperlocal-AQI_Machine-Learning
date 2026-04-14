import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GRID_CSV = Path("outputs") / "bengaluru_grid_1000m_centroids.csv"
DEFAULT_STATIONS_CSV = Path("data") / "metadata" / "bangalore_stations.csv"
DEFAULT_PROCESSED_CSV = Path("data") / "processed" / "bangalore_processed.csv"
DEFAULT_DISTANCE_OUT = Path("data") / "processed" / "bengaluru_grid_station_distance_features.csv"
DEFAULT_HOURLY_OUT = Path("data") / "processed" / "bengaluru_grid_hourly_feature_table.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for distance feature generation."""
    parser = argparse.ArgumentParser(
        description="Build grid-level station-distance features and an hourly grid feature table."
    )
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV, help="Grid centroid CSV path.")
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=DEFAULT_STATIONS_CSV,
        help="Station metadata CSV path.",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=DEFAULT_PROCESSED_CSV,
        help="Processed hourly Bengaluru dataset path.",
    )
    parser.add_argument(
        "--distance-out",
        type=Path,
        default=DEFAULT_DISTANCE_OUT,
        help="Output CSV for grid-level station-distance features.",
    )
    parser.add_argument(
        "--hourly-out",
        type=Path,
        default=DEFAULT_HOURLY_OUT,
        help="Output CSV for hourly grid feature table.",
    )
    return parser.parse_args()


def load_grid_and_station_data(grid_csv: Path, stations_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load grid centroids and station metadata."""
    if not grid_csv.exists():
        raise FileNotFoundError(f"Grid CSV not found: {grid_csv}")
    if not stations_csv.exists():
        raise FileNotFoundError(f"Station metadata CSV not found: {stations_csv}")

    grid_df = pd.read_csv(grid_csv)
    stations_df = pd.read_csv(stations_csv)

    required_grid_cols = {"grid_id", "centroid_lat", "centroid_lon"}
    required_station_cols = {"station", "latitude", "longitude"}

    missing_grid_cols = required_grid_cols - set(grid_df.columns)
    missing_station_cols = required_station_cols - set(stations_df.columns)

    if missing_grid_cols:
        raise ValueError(f"Grid CSV missing required columns: {missing_grid_cols}")
    if missing_station_cols:
        raise ValueError(f"Station CSV missing required columns: {missing_station_cols}")

    return grid_df, stations_df


def load_hourly_timestamps(processed_csv: Path) -> pd.DataFrame:
    """Load unique hourly timestamps from the processed CAAQMS dataset."""
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_csv}. Run scripts/run_pipeline.py first."
        )

    processed_df = pd.read_csv(processed_csv, usecols=["timestamp"])
    processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"])

    timestamps_df = processed_df[["timestamp"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    timestamps_df["hour"] = timestamps_df["timestamp"].dt.hour
    timestamps_df["month"] = timestamps_df["timestamp"].dt.month

    return timestamps_df


def haversine_distance_matrix(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
) -> np.ndarray:
    """Compute great-circle distance matrix in kilometers between grid points and stations."""
    earth_radius_km = 6371.0

    grid_lat_rad = np.radians(grid_lats)[:, np.newaxis]
    grid_lon_rad = np.radians(grid_lons)[:, np.newaxis]
    station_lat_rad = np.radians(station_lats)[np.newaxis, :]
    station_lon_rad = np.radians(station_lons)[np.newaxis, :]

    dlat = station_lat_rad - grid_lat_rad
    dlon = station_lon_rad - grid_lon_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(grid_lat_rad) * np.cos(station_lat_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def sanitize_station_name(station_name: str) -> str:
    """Convert a station name into a column-safe suffix."""
    return re.sub(r"[^a-z0-9]+", "_", station_name.lower()).strip("_")


def compute_grid_station_distance_features(grid_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Compute nearest-station and all-station distance features for each grid cell."""
    distance_matrix = haversine_distance_matrix(
        grid_df["centroid_lat"].to_numpy(),
        grid_df["centroid_lon"].to_numpy(),
        stations_df["latitude"].to_numpy(),
        stations_df["longitude"].to_numpy(),
    )

    distance_features = grid_df.copy()

    for station_idx, station_row in stations_df.reset_index(drop=True).iterrows():
        station_col = f"dist_to_{sanitize_station_name(station_row['station'])}_km"
        distance_features[station_col] = distance_matrix[:, station_idx]

    sorted_indices = np.argsort(distance_matrix, axis=1)
    nearest_indices = sorted_indices[:, 0]
    second_nearest_indices = sorted_indices[:, 1]

    stations_array = stations_df["station"].to_numpy()
    row_indices = np.arange(len(distance_features))

    distance_features["nearest_station"] = stations_array[nearest_indices]
    distance_features["nearest_station_distance_km"] = distance_matrix[row_indices, nearest_indices]
    distance_features["second_nearest_station"] = stations_array[second_nearest_indices]
    distance_features["second_nearest_station_distance_km"] = distance_matrix[row_indices, second_nearest_indices]
    distance_features["mean_station_distance_km"] = distance_matrix.mean(axis=1)
    distance_features["max_station_distance_km"] = distance_matrix.max(axis=1)
    distance_features["stations_within_5km"] = (distance_matrix <= 5).sum(axis=1)
    distance_features["stations_within_10km"] = (distance_matrix <= 10).sum(axis=1)

    ordered_columns = [
        "grid_id",
        "centroid_lat",
        "centroid_lon",
        "nearest_station",
        "nearest_station_distance_km",
        "second_nearest_station",
        "second_nearest_station_distance_km",
        "mean_station_distance_km",
        "max_station_distance_km",
        "stations_within_5km",
        "stations_within_10km",
    ]
    distance_columns = [col for col in distance_features.columns if col.startswith("dist_to_")]

    return distance_features[ordered_columns + distance_columns]


def build_hourly_feature_table(
    grid_distance_df: pd.DataFrame,
    timestamps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create an hourly grid feature table by crossing timestamps with grid features."""
    hourly_feature_df = timestamps_df.assign(_merge_key=1).merge(
        grid_distance_df.assign(_merge_key=1),
        on="_merge_key",
        how="inner",
    )
    hourly_feature_df = hourly_feature_df.drop(columns="_merge_key")

    ordered_cols = ["timestamp", "hour", "month", "grid_id", "centroid_lat", "centroid_lon"]
    other_cols = [col for col in hourly_feature_df.columns if col not in ordered_cols]
    return hourly_feature_df[ordered_cols + other_cols]


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    grid_df, stations_df = load_grid_and_station_data(args.grid_csv, args.stations_csv)
    timestamps_df = load_hourly_timestamps(args.processed_csv)

    grid_distance_df = compute_grid_station_distance_features(grid_df, stations_df)
    hourly_feature_df = build_hourly_feature_table(grid_distance_df, timestamps_df)

    save_dataframe(grid_distance_df, args.distance_out)
    save_dataframe(hourly_feature_df, args.hourly_out)

    print(f"Grid distance features saved to {args.distance_out}")
    print(f"Hourly grid feature table saved to {args.hourly_out}")
    print(f"Grid cells: {len(grid_distance_df)}")
    print(f"Unique hourly timestamps: {len(timestamps_df)}")
    print(f"Hourly feature rows: {len(hourly_feature_df)}")


if __name__ == "__main__":
    main()
