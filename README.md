# Hyperlocal AQI Machine Learning

This project explores how to build a hyperlocal AQI monitoring framework using existing regulatory CAAQMS stations and auxiliary features, without assuming a dense low-cost sensor network from the beginning.

The long-term goal is two-stage:

1. infer fine-scale PM2.5 and AQI patterns from sparse but reliable regulatory stations using features such as weather, temporal signals, and later satellite or other spatial datasets
2. use model error, spatial gaps, and uncertainty to decide where a minimal number of low-cost sensors should be placed for efficient sensor deployment

## Why Bengaluru

Bengaluru is the current pilot city because it has a relatively good number of CAAQMS stations and is a large tier-1 city for which additional datasets can later be integrated. Other cities are intended to be added later, in decreasing order of dataset availability and quality.

## Current Status

This repository is currently in the baseline experimentation stage, not the final hyperlocal AQI system stage.

What is already implemented:

- a preprocessing pipeline for the full Bengaluru raw station dataset
- reproducible station selection based on row-count coverage
- processed dataset generation for the selected stations
- validation summaries for processed data quality
- spatial holdout evaluation using three test setups
- baseline comparison across five regression models

What is not yet implemented:

- city-wide hyperlocal prediction on a dense grid or neighborhood map
- richer auxiliary features such as satellite, land-use, traffic, or road-network variables
- uncertainty-aware prediction
- sensor placement optimization
- multi-city expansion

## Bengaluru Workflow

The current workflow is:

1. start from the complete raw Bengaluru CAAQMS CSV in `data/raw/bangalore_caaqms/bangalore_caaqms_2025_combined.csv`
2. rank all stations by row count
3. select the top 6 stations for the current pilot study
4. clean and standardize the selected-station data
5. add temporal features and station coordinates
6. save the final modeling dataset to `data/processed/bangalore_processed.csv`
7. run baseline models under spatial holdout evaluation
8. export comparison CSVs for result analysis

## Station Selection

The full Bengaluru raw dataset contains all available stations. For the current study, 6 stations are selected from the original 14 based on total row count in the combined raw CSV. This is being used as a simple reliability and continuity proxy for the pilot phase.

The selected stations are stored in `data/metadata/bangalore_stations.csv` and are used to attach latitude and longitude to the processed dataset.

## Evaluation Design

The project currently evaluates model performance using station-drop experiments that simulate sparse sensor availability:

- `Test A`: drop the most central station and predict it from the rest
- `Test B`: drop the most peripheral station and predict it from the rest
- `Test C`: leave-one-station-out across all selected stations

This setup is meant to test how well the model generalizes spatially when station support changes.

## Models Compared So Far

The current baseline models are:

- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

These are being compared under the same spatial evaluation mechanism so that model differences can be separated from spatial sampling effects.

## Current Features

The present baseline feature set is intentionally simple:

- temperature
- humidity
- wind speed
- wind direction
- hour
- month

The current target variable is `pm25`.

## Repository Structure

- `data/raw/`: raw input datasets
- `data/metadata/`: station metadata such as latitude and longitude
- `data/processed/`: final processed datasets used for modeling
- `src/ingestion/`: raw data loading and standardization
- `src/cleaning/`: cleaning and interpolation logic
- `src/features/`: feature engineering logic
- `src/evaluation/`: spatial split logic and evaluation metrics
- `scripts/`: runnable project scripts
- `outputs/`: saved model comparison results
- `notebooks/`: exploratory notebook work from earlier phases

## Main Scripts

- `scripts/run_pipeline.py`
  Builds the processed Bengaluru dataset from the full raw file by selecting the top 6 stations, cleaning the data, adding time features, and attaching coordinates.

- `scripts/validate_processed_data.py`
  Reads the raw and processed datasets and reports row counts, station coverage, temporal coverage, missing values, and completeness summaries.

- `scripts/run_baseline_tests.py`
  Runs the Random Forest baseline under the three evaluation tests.

- `scripts/run_baseline_linear.py`
- `scripts/run_baseline_xgboost.py`
- `scripts/run_baseline_lightgbm.py`
- `scripts/run_baseline_catboost.py`
  Run the other baseline models under the same evaluation setup.

- `scripts/generate_comparison_csvs.py`
  Combines baseline output files and writes per-test comparison tables.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the processed dataset:

```bash
python3 scripts/run_pipeline.py
```

Validate the processed dataset:

```bash
python3 scripts/validate_processed_data.py
```

Run baseline experiments:

```bash
python3 scripts/run_baseline_tests.py
python3 scripts/run_baseline_linear.py
python3 scripts/run_baseline_xgboost.py
python3 scripts/run_baseline_lightgbm.py
python3 scripts/run_baseline_catboost.py
```

Generate comparison CSVs:

```bash
python3 scripts/generate_comparison_csvs.py
```

## Important Interpretation Note

At the moment, this repository should be read as a baseline research pipeline, not as a finished hyperlocal AQI product.

The current results are useful for understanding whether sparse-station inference is promising, but they do not yet constitute a fully operational city-scale hyperlocal AQI system.

## Next Development Goals

The next major steps are:

1. add richer auxiliary datasets such as satellite, land-use, traffic, and spatial context features
2. move from station-level holdout prediction to city-wide hyperlocal prediction units
3. estimate uncertainty or identify weak-coverage spatial regions
4. use those gaps to design efficient low-cost sensor placement
5. later extend the framework to additional cities
