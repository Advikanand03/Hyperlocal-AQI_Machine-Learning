import pandas as pd
import logging

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features: hour and month from timestamp."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    df = df.copy()  # avoid modifying original
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    logger.info("Temporal features added: hour and month")

    return df