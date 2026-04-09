from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.station_drop import (
    test_A_central,
    test_B_peripheral,
    test_C_loso
)


def main():
    root = Path(__file__).resolve().parents[1]

    data = pd.read_csv(root / "data/processed/bangalore_processed.csv")
    stations = pd.read_csv(root / "data/metadata/bangalore_stations.csv")

    results = []

    results.append(test_A_central(data, stations))
    results.append(test_B_peripheral(data, stations))

    loso = test_C_loso(data)
    results_df = pd.concat([pd.DataFrame(results), loso], ignore_index=True)

    results_df.to_csv(root / "outputs/baseline_results.csv", index=False)
    print("✅ Baseline tests A, B, C completed")


if __name__ == "__main__":
    main()