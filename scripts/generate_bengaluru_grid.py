import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from shapely.geometry import box


DEFAULT_PLACE = "Bengaluru, India"
DEFAULT_RESOLUTION_M = 1000


def fetch_city_boundary(place_name: str = DEFAULT_PLACE) -> gpd.GeoDataFrame:
    """Fetch the city boundary in WGS84 coordinates."""
    city_gdf = ox.geocode_to_gdf(place_name)
    if city_gdf.empty:
        raise ValueError(f"No boundary returned for place: {place_name}")
    return city_gdf[["geometry"]].copy()


def project_to_metric_crs(city_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project a boundary GeoDataFrame to a suitable local metric CRS."""
    metric_crs = city_gdf.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Could not estimate a metric CRS for the city boundary")
    return city_gdf.to_crs(metric_crs)


def create_square_grid(boundary_gdf: gpd.GeoDataFrame, resolution_m: int) -> gpd.GeoDataFrame:
    """Create a uniform square grid covering the boundary extent."""
    minx, miny, maxx, maxy = boundary_gdf.total_bounds

    x_coords = np.arange(minx, maxx + resolution_m, resolution_m)
    y_coords = np.arange(miny, maxy + resolution_m, resolution_m)

    cells = [
        box(x0, y0, x0 + resolution_m, y0 + resolution_m)
        for x0 in x_coords[:-1]
        for y0 in y_coords[:-1]
    ]

    return gpd.GeoDataFrame({"geometry": cells}, crs=boundary_gdf.crs)


def clip_grid_to_boundary(grid_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip the grid to the city boundary and keep only intersecting cells."""
    clipped = gpd.clip(grid_gdf, boundary_gdf)
    clipped = clipped[~clipped.geometry.is_empty].copy()
    clipped.reset_index(drop=True, inplace=True)
    return clipped


def add_grid_metadata(grid_gdf: gpd.GeoDataFrame, place_prefix: str = "BLR") -> gpd.GeoDataFrame:
    """Assign unique grid IDs and compute centroid latitude/longitude in WGS84."""
    grid_gdf = grid_gdf.copy()
    grid_gdf["grid_id"] = [f"{place_prefix}_{idx:05d}" for idx in range(1, len(grid_gdf) + 1)]

    centroids_metric = grid_gdf.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_metric, crs=grid_gdf.crs).to_crs(epsg=4326)

    grid_gdf["centroid_lon"] = centroids_wgs84.x
    grid_gdf["centroid_lat"] = centroids_wgs84.y

    return grid_gdf[["grid_id", "geometry", "centroid_lat", "centroid_lon"]]


def save_grid_outputs(
    grid_gdf: gpd.GeoDataFrame,
    geojson_path: Path,
    csv_path: Path,
) -> None:
    """Save grid polygons as GeoJSON and centroid coordinates as CSV."""
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    grid_wgs84 = grid_gdf.to_crs(epsg=4326)
    grid_wgs84.to_file(geojson_path, driver="GeoJSON")
    grid_wgs84[["grid_id", "centroid_lat", "centroid_lon"]].to_csv(csv_path, index=False)


def plot_boundary_and_grid(
    city_boundary_wgs84: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    plot_path: Path | None = None,
) -> None:
    """Plot the city boundary and generated grid overlay."""
    fig, ax = plt.subplots(figsize=(10, 10))

    grid_wgs84 = grid_gdf.to_crs(epsg=4326)
    grid_wgs84.boundary.plot(ax=ax, linewidth=0.4, color="steelblue", alpha=0.8)
    city_boundary_wgs84.boundary.plot(ax=ax, linewidth=1.5, color="black")

    ax.set_title("Bengaluru Grid Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()


def generate_bengaluru_grid(
    place_name: str = DEFAULT_PLACE,
    resolution_m: int = DEFAULT_RESOLUTION_M,
    geojson_path: Path | None = None,
    csv_path: Path | None = None,
    plot_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Generate a clipped grid over Bengaluru and optionally save outputs."""
    city_boundary_wgs84 = fetch_city_boundary(place_name)
    city_boundary_metric = project_to_metric_crs(city_boundary_wgs84)

    grid_metric = create_square_grid(city_boundary_metric, resolution_m)
    clipped_grid_metric = clip_grid_to_boundary(grid_metric, city_boundary_metric)
    final_grid = add_grid_metadata(clipped_grid_metric)

    if geojson_path is not None and csv_path is not None:
        save_grid_outputs(final_grid, geojson_path, csv_path)

    plot_boundary_and_grid(city_boundary_wgs84, final_grid, plot_path=plot_path)
    return final_grid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a clipped grid over Bengaluru.")
    parser.add_argument("--place", default=DEFAULT_PLACE, help="Place name to geocode.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION_M,
        help="Grid resolution in meters.",
    )
    parser.add_argument(
        "--geojson-out",
        type=Path,
        default=None,
        help="Optional output path for GeoJSON grid polygons.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output path for centroid CSV.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional output path for the plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resolution_tag = f"{args.resolution}m"
    geojson_path = args.geojson_out or Path("outputs") / f"bengaluru_grid_{resolution_tag}.geojson"
    csv_path = args.csv_out or Path("outputs") / f"bengaluru_grid_{resolution_tag}_centroids.csv"
    plot_path = args.plot_out or Path("outputs") / f"bengaluru_grid_{resolution_tag}.png"

    grid_gdf = generate_bengaluru_grid(
        place_name=args.place,
        resolution_m=args.resolution,
        geojson_path=geojson_path,
        csv_path=csv_path,
        plot_path=plot_path,
    )

    print(f"Generated {len(grid_gdf)} grid cells at {args.resolution} m resolution")
    print(f"Saved GeoJSON to {geojson_path}")
    print(f"Saved centroid CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
