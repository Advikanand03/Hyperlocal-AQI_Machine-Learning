import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluation.metrics import regression_metrics
from src.evaluation.spatial_splits import (
    identify_central_station,
    identify_peripheral_station
)


FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "hour",
    "month"
]

TARGET = "pm25"


def run_station_drop_with_model(df, test_station, model):
    train = df[df.station != test_station]
    test = df[df.station == test_station]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return regression_metrics(y_test, preds)


def run_station_drop(df, test_station):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    return run_station_drop_with_model(df, test_station, model)


def test_A_central(df, station_meta, model=None):
    station = identify_central_station(station_meta)
    model_obj = model if model is not None else RandomForestRegressor(n_estimators=200, random_state=42)
    metrics = run_station_drop_with_model(df, station, model_obj)
    return {"test": "A_central", "station": station, "model": model_obj.__class__.__name__, **metrics}


def test_B_peripheral(df, station_meta, model=None):
    station = identify_peripheral_station(station_meta)
    model_obj = model if model is not None else RandomForestRegressor(n_estimators=200, random_state=42)
    metrics = run_station_drop_with_model(df, station, model_obj)
    return {"test": "B_peripheral", "station": station, "model": model_obj.__class__.__name__, **metrics}


def test_C_loso(df, model=None):
    results = []
    for station in sorted(df.station.unique()):
        model_obj = model if model is not None else RandomForestRegressor(n_estimators=200, random_state=42)
        metrics = run_station_drop_with_model(df, station, model_obj)
        results.append({"test": "C_LOSO", "station": station, "model": model_obj.__class__.__name__, **metrics})
    return pd.DataFrame(results)