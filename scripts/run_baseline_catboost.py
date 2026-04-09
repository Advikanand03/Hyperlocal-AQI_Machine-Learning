import sys
from pathlib import Path

import pandas as pd

try:
    from catboost import CatBoostRegressor
except ImportError as e:
    raise ImportError("catboost must be installed to run this script") from e

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.station_drop import (
    test_A_central,
    test_B_peripheral,
    test_C_loso,
)


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "processed" / "bangalore_processed.csv")
    stations = pd.read_csv(root / "data" / "metadata" / "bangalore_stations.csv")

    model = CatBoostRegressor(iterations=500, random_state=42, verbose=False)
    results = [
        test_A_central(df, stations, model),
        test_B_peripheral(df, stations, model),
    ]
    loso = test_C_loso(df, model)

    results_df = pd.concat([pd.DataFrame(results), loso], ignore_index=True)
    out_path = root / "outputs" / "baseline_catboost.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("✅ CatBoost baseline completed ->", out_path)


if __name__ == "__main__":
    main()
