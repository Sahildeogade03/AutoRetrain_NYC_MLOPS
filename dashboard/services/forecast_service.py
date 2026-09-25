from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONITORING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "historical_monitoring_forecast.parquet"
)


def load_monitoring_forecast() -> pd.DataFrame:
    """Load the persisted historical forecast/actual monitoring data."""

    if not MONITORING_PATH.exists():
        raise FileNotFoundError(
            f"Monitoring forecast file not found: {MONITORING_PATH}"
        )

    df = pd.read_parquet(MONITORING_PATH)

    required_columns = {
        "timestamp",
        "actual_demand",
        "forecast_demand",
        "PULocationID",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in monitoring data: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.sort_values("timestamp")


def get_system_forecast() -> pd.DataFrame:
    """
    Aggregate zone-level forecasts into system-wide hourly demand.
    """

    df = load_monitoring_forecast()

    hourly = (
        df.groupby("timestamp")
        .agg(
            actual_demand=("actual_demand", "sum"),
            forecast_demand=("forecast_demand", "sum"),
        )
        .sort_index()
        .reset_index()
    )

    hourly["residual"] = (
        hourly["actual_demand"]
        - hourly["forecast_demand"]
    )

    return hourly


def get_forecast_metrics() -> dict:
    """Calculate system-wide forecast metrics."""

    hourly = get_system_forecast()

    residual = (
        hourly["actual_demand"]
        - hourly["forecast_demand"]
    )

    mae = residual.abs().mean()

    rmse = np.sqrt(
        np.mean(residual ** 2)
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "forecast_data": hourly,
        "latest_timestamp": hourly["timestamp"].max(),
        "observation_count": len(hourly),
    }