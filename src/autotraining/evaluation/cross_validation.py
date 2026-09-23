# src/autotraining/evaluation/cross_validation.py

from __future__ import annotations

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.metrics import mae, rmse, smape


def make_walk_forward_folds(
    total_length: int,
    min_train_size: int,
    horizon: int,
    stride: int,
) -> list[tuple[int, int]]:
    """
    Create expanding-window walk-forward validation folds.

    Each fold is represented as:
        (train_end, test_end)

    The training window always starts at index 0 and expands
    as validation progresses.
    """

    folds = []

    train_end = min_train_size

    while train_end + horizon <= total_length:
        folds.append(
            (train_end, train_end + horizon)
        )

        train_end += stride

    return folds


def _compute_mape(
    actual: TimeSeries,
    predicted: TimeSeries,
) -> float:
    """
    Compute MAPE while excluding zero actual values.
    """

    actual_values = actual.values(copy=False).squeeze()
    predicted_values = predicted.values(copy=False).squeeze()

    nonzero_mask = actual_values != 0

    if not nonzero_mask.any():
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual_values[nonzero_mask]
                    - predicted_values[nonzero_mask]
                )
                / actual_values[nonzero_mask]
            )
        )
        * 100
    )


def compute_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
) -> dict[str, float]:
    """
    Compute forecasting metrics.

    Metrics:
        MAE
        RMSE
        MAPE (excluding zero actuals)
        sMAPE
    """

    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": _compute_mape(actual, predicted),
        "smape": smape(actual, predicted),
    }


def _clip_nonnegative(
    series: TimeSeries,
) -> TimeSeries:
    """
    Clip demand forecasts at zero.
    """

    return series.map(
        lambda x: np.maximum(x, 0.0)
    )


def _fit_predict_naive(
    train_series: list[TimeSeries],
    horizon: int,
    seasonal_k: int,
) -> list[TimeSeries]:
    """
    Generate forecasts using the seasonal naive baseline.
    """

    from darts.models import NaiveSeasonal

    predictions = []

    for series in train_series:

        model = NaiveSeasonal(
            K=seasonal_k
        )

        model.fit(series)

        prediction = model.predict(
            horizon
        )

        predictions.append(
            _clip_nonnegative(prediction)
        )

    return predictions


def _fit_predict_lightgbm_raw(
    train_series: list[TimeSeries],
    horizon: int,
    future_covariates: TimeSeries,
    build_model,
) -> list[TimeSeries]:
    """
    Train and forecast the global LightGBM model
    on raw demand.
    """

    model = build_model(horizon)

    predictions = model.predict(
        n=horizon,
        series=train_series,
        future_covariates=future_covariates,
    )

    return [
        _clip_nonnegative(prediction)
        for prediction in predictions
    ]


def run_walk_forward_cv(
    raw_series: list[TimeSeries],
    future_covariates: TimeSeries,
    folds: list[tuple[int, int]],
    model_name: str,
    model_cfg: dict,
    seasonal_k: int,
) -> pd.DataFrame:
    """
    Run walk-forward cross-validation for one model.

    Supported model kinds:

        naive
        global_raw

    Returns:
        One row per fold × zone.
    """

    rows = []

    for fold_id, (train_end, test_end) in enumerate(
        folds,
        start=1,
    ):

        print(
            f"Fold {fold_id}/{len(folds)} | "
            f"train_end={train_end} | "
            f"test_end={test_end}"
        )

        # --------------------------------------------------
        # Train / test split
        # --------------------------------------------------

        train_series = [
            series[:train_end]
            for series in raw_series
        ]

        test_series = [
            series[train_end:test_end]
            for series in raw_series
        ]

        horizon = test_end - train_end

        # --------------------------------------------------
        # Generate predictions
        # --------------------------------------------------

        if model_cfg["kind"] == "naive":

            predictions = _fit_predict_naive(
                train_series=train_series,
                horizon=horizon,
                seasonal_k=seasonal_k,
            )

        elif model_cfg["kind"] == "global_raw":

            predictions = _fit_predict_lightgbm_raw(
                train_series=train_series,
                horizon=horizon,
                future_covariates=future_covariates,
                build_model=model_cfg["builder"],
            )

        else:

            raise ValueError(
                f"Unsupported model kind: "
                f"{model_cfg['kind']}"
            )

        # --------------------------------------------------
        # Evaluate each zone
        # --------------------------------------------------

        for zone_idx, (
            actual,
            prediction,
        ) in enumerate(
            zip(
                test_series,
                predictions,
            )
        ):

            metrics = compute_metrics(
                actual,
                prediction,
            )

            rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "zone_idx": zone_idx,
                    "train_end": train_end,
                    "test_end": test_end,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)