import pandas as pd
import numpy as np
from typing import Tuple


def compute_centroid(stations: pd.DataFrame) -> Tuple[float, float]:
    """Return (lat, lon) centroid of all stations."""
    return stations["latitude"].mean(), stations["longitude"].mean()


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def identify_central_station(stations: pd.DataFrame) -> str:
    """Station closest to centroid."""
    clat, clon = compute_centroid(stations)
    dists = stations.apply(
        lambda r: haversine(r.latitude, r.longitude, clat, clon), axis=1
    )
    return stations.loc[dists.idxmin(), "station"]


def identify_peripheral_station(stations: pd.DataFrame) -> str:
    """Station with largest mean distance to others."""
    coords = stations[["latitude", "longitude"]].values
    dist_matrix = np.array([
        [haversine(*coords[i], *coords[j]) for j in range(len(coords))]
        for i in range(len(coords))
    ])
    mean_dist = dist_matrix.mean(axis=1)
    return stations.iloc[mean_dist.argmax()]["station"]