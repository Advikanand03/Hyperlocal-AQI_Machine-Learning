import os
from pathlib import Path
import cdsapi
from dotenv import load_dotenv

# ==============================
# LOAD ENV VARIABLES
# ==============================
load_dotenv()

CDS_API_URL = os.getenv("CDS_API_URL")
CDS_API_KEY = os.getenv("CDS_API_KEY")

# ==============================
# CONFIG
# ==============================
YEAR = "2025"
OUTPUT_DIR = Path("/Users/barkhaanand/Hyperlocal-AQI_Machine-Learning/data/era5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = [13.2, 77.3, 12.7, 77.7]

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation"
]

# ==============================
# DOWNLOAD FUNCTION
# ==============================
def download_all_months():
    client = cdsapi.Client(
        url=CDS_API_URL,
        key=CDS_API_KEY
    )

    for m in range(1, 13):
        month = f"{m:02d}"
        output_file = OUTPUT_DIR / f"era5_land_{YEAR}_{month}.grib"

        print(f"\nDownloading {YEAR}-{month}...")

        request = {
            "variable": VARIABLES,
            "year": YEAR,
            "month": month,
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "area": AREA,
            "format": "grib"
        }

        client.retrieve(
            "reanalysis-era5-land",
            request,
            str(output_file)
        )

        print(f"Saved: {output_file}")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    download_all_months()