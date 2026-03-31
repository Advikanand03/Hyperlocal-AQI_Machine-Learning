import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.load_cpcb import load_cpcb
from src.cleaning.clean_data import clean_data
from src.features.temporal import add_temporal_features


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    base = Path(__file__).resolve().parents[1]
    raw_path = base / "data" / "raw" / "bangalore_caaqms" / "bangalore_caaqms_2025_combined.csv"
    processed_path = base / "data" / "processed" / "bangalore_processed.csv"

    df = load_cpcb(raw_path)
    df = clean_data(df)
    df = add_temporal_features(df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    logging.info(f"Pipeline completed. Final data saved to {processed_path}")


if __name__ == "__main__":
    main()