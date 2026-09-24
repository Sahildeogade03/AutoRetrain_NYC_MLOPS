"""Module scaffold: inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel

from autotraining.features.demand_features import (
    build_calendar_covariates,
)


MODEL_PATH = Path("models/lightgbm_raw_v2.pt")
PANEL_PATH = Path("data/processed/zone_panel.parquet")


def load_model(
    model_path: Path = MODEL_PATH,
) -> LightGBMModel:
    """Load the validated production LightGBM model."""

    return LightGBMModel.load(
        str(model_path)
    )


def load_zone_panel(
    panel_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    """Load the production zone-demand panel."""

    panel = pd.read_parquet(
        panel_path
    )

    panel.index = pd.to_datetime(
        panel.index
    )

    panel = panel.sort_index()

    return panel


def build_zone_series(
    panel: pd.DataFrame,
) -> list[TimeSeries]:
    """Convert the zone panel into Darts time series."""

    return [
        TimeSeries.from_series(
            panel[zone]
        )
        for zone in panel.columns
    ]


def build_future_covariates(
    index: pd.DatetimeIndex,
) -> TimeSeries:
    """Build calendar covariates for inference."""

    return build_calendar_covariates(
        index
    )


def forecast_future(
    horizon: int = 168,
    model_path: Path = MODEL_PATH,
    panel_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    """
    Generate future forecasts from the latest available data.

    No model training occurs here.
    """

    panel = load_zone_panel(
        panel_path
    )

    model = load_model(
        model_path
    )

    zone_series = build_zone_series(
        panel
    )

    last_timestamp = panel.index[-1]

    future_index = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )

    covariate_index = pd.date_range(
        start=panel.index[0],
        end=future_index[-1],
        freq="h",
    )

    calendar_covariates = (
        build_future_covariates(
            covariate_index
        )
    )

    future_covariates = [
        calendar_covariates
        for _ in zone_series
    ]

    forecasts = model.predict(
        n=horizon,
        series=zone_series,
        future_covariates=future_covariates,
    )

    return _forecasts_to_dataframe(
        forecasts
    )


def forecast_historical(
    forecast_start: str,
    horizon: int,
    model_path: Path = MODEL_PATH,
    panel_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    """
    Generate historical forecasts without retraining.

    The forecast begins at `forecast_start` and uses only
    observations available before that timestamp.
    """

    panel = load_zone_panel(
        panel_path
    )

    model = load_model(
        model_path
    )

    forecast_start = pd.Timestamp(
        forecast_start
    )

    if forecast_start <= panel.index[0]:
        raise ValueError(
            "forecast_start must be after "
            "the beginning of the panel."
        )

    history = panel[
        panel.index < forecast_start
    ]

    if history.empty:
        raise ValueError(
            "No historical observations are "
            "available before forecast_start."
        )

    zone_series = build_zone_series(
        history
    )

    forecast_end = (
        forecast_start
        + pd.Timedelta(hours=horizon - 1)
    )

    covariate_index = pd.date_range(
        start=history.index[0],
        end=forecast_end,
        freq="h",
    )

    calendar_covariates = (
        build_future_covariates(
            covariate_index
        )
    )

    future_covariates = [
        calendar_covariates
        for _ in zone_series
    ]

    forecasts = model.predict(
        n=horizon,
        series=zone_series,
        future_covariates=future_covariates,
    )

    forecast_df = _forecasts_to_dataframe(
        forecasts
    )

    actual = panel.loc[
        forecast_start:forecast_end
    ].copy()

    actual.index.name = "timestamp"

    actual_df = (
        actual
        .stack()
        .rename("actual_demand")
        .reset_index()
    )

    actual_df = actual_df.rename(
        columns={
            "level_1": "PULocationID"
        }
    )

    result = forecast_df.merge(
        actual_df,
        on=[
            "timestamp",
            "PULocationID",
        ],
        how="inner",
    )

    result["residual"] = (
        result["actual_demand"]
        - result["forecast_demand"]
    )

    return result


def _forecasts_to_dataframe(
    forecasts: list[TimeSeries],
) -> pd.DataFrame:
    """Convert Darts forecasts to the project's tabular format."""

    frames = []

    for series in forecasts:

        zone_id = series.components[0]

        frame = series.to_dataframe(
            copy=True
        ).reset_index()

        frame = frame.rename(
            columns={
                frame.columns[0]: "timestamp",
                frame.columns[1]: "forecast_demand",
            }
        )

        frame["PULocationID"] = int(
            zone_id
        )

        frames.append(
            frame[
                [
                    "timestamp",
                    "forecast_demand",
                    "PULocationID",
                ]
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )