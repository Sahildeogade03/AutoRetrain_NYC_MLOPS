from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel

from autotraining.features.demand_features import (
    build_calendar_covariates,
)


DEFAULT_HOLDOUT_HORIZON = 336


def evaluate_candidate(
    model: LightGBMModel,
    panel: pd.DataFrame,
    holdout_horizon: int = DEFAULT_HOLDOUT_HORIZON,
) -> dict:
    """
    Evaluate a candidate model on a strictly held-out
    final time window.

    The candidate must already have been trained only
    on observations before the holdout period.
    """

    if len(panel) <= holdout_horizon:
        raise ValueError(
            "Panel is too short for the requested holdout."
        )

    train_panel = panel.iloc[:-holdout_horizon]
    holdout_panel = panel.iloc[-holdout_horizon:]

    zone_series = [
        TimeSeries.from_series(
            train_panel[zone]
        )
        for zone in train_panel.columns
    ]

    forecast_end = holdout_panel.index[-1]

    covariate_index = pd.date_range(
        start=train_panel.index[0],
        end=forecast_end,
        freq="h",
    )

    calendar_covariates = build_calendar_covariates(
        covariate_index
    )

    future_covariates = [
        calendar_covariates
        for _ in zone_series
    ]

    forecasts = model.predict(
        n=holdout_horizon,
        series=zone_series,
        future_covariates=future_covariates,
    )

    zone_results = []

    for zone, forecast in zip(
        holdout_panel.columns,
        forecasts,
    ):

        actual = holdout_panel[zone].to_numpy()

        predicted = (
            forecast
            .pd_series()
            .to_numpy()
        )

        mae = float(
            np.mean(
                np.abs(
                    actual - predicted
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (actual - predicted) ** 2
                )
            )
        )

        zone_results.append(
            {
                "zone": int(zone),
                "mae": mae,
                "rmse": rmse,
            }
        )

    zone_metrics = pd.DataFrame(
        zone_results
    )

    return {
        "model": "lightgbm_raw_candidate",
        "holdout_horizon": holdout_horizon,
        "holdout_start": str(
            holdout_panel.index[0]
        ),
        "holdout_end": str(
            holdout_panel.index[-1]
        ),
        "mae": float(
            zone_metrics["mae"].mean()
        ),
        "rmse": float(
            zone_metrics["rmse"].mean()
        ),
        "zones": int(
            zone_metrics["zone"].nunique()
        ),
        "zone_metrics": zone_metrics,
    }