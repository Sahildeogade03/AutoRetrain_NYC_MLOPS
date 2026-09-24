"""Module scaffold: performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt


def calculate_residuals(
    actual: pd.Series,
    forecast: pd.Series,
) -> pd.Series:
    """
    Calculate forecast residuals.

    residual = actual - forecast
    """

    aligned = pd.concat(
        [
            actual.rename("actual"),
            forecast.rename("forecast"),
        ],
        axis=1,
    ).dropna()

    residuals = aligned["actual"] - aligned["forecast"]
    residuals.name = "residual"

    return residuals


def calculate_mae(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate mean absolute error."""

    aligned = pd.concat(
        [
            actual.rename("actual"),
            forecast.rename("forecast"),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError("No overlapping actual/forecast observations.")

    return float(
        np.mean(
            np.abs(
                aligned["actual"] - aligned["forecast"]
            )
        )
    )


def detect_change_points(
    residuals: pd.Series,
    model: str = "l2",
    penalty: float = 10.0,
    min_size: int = 24,
) -> list[int]:
    """
    Detect structural change points in forecast residuals.

    Parameters
    ----------
    residuals:
        Forecast residual time series.

    model:
        Cost function used by ruptures.

    penalty:
        Penalty controlling the number of detected change points.

    min_size:
        Minimum segment length in observations.

    Returns
    -------
    list[int]
        Change-point indices.
    """

    values = residuals.dropna().to_numpy(dtype=float)

    if len(values) < 2 * min_size:
        return []

    detector = rpt.Pelt(
        model=model,
        min_size=min_size,
        jump=1,
    )

    change_points = detector.fit(values).predict(
        pen=penalty
    )

    # ruptures includes the final endpoint.
    return [
        cp
        for cp in change_points
        if cp < len(values)
    ]


def evaluate_residual_drift(
    residuals: pd.Series,
    penalty: float = 10.0,
    min_size: int = 24,
) -> dict:
    """
    Evaluate residual drift using change-point detection.

    Both residual bias and absolute residual magnitude
    are monitored.
    """

    residuals = residuals.dropna()

    if residuals.empty:
        raise ValueError("Residual series is empty.")

    bias_change_points = detect_change_points(
        residuals=residuals,
        penalty=penalty,
        min_size=min_size,
    )

    magnitude_change_points = detect_change_points(
        residuals=residuals.abs(),
        penalty=penalty,
        min_size=min_size,
    )

    latest_bias_cp = (
        bias_change_points[-1]
        if bias_change_points
        else None
    )

    latest_magnitude_cp = (
        magnitude_change_points[-1]
        if magnitude_change_points
        else None
    )

    return {
        "drift_detected": bool(
            bias_change_points
            or magnitude_change_points
        ),
        "bias_change_points": bias_change_points,
        "magnitude_change_points": magnitude_change_points,
        "latest_bias_change_point": latest_bias_cp,
        "latest_magnitude_change_point": latest_magnitude_cp,
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "mean_absolute_error": float(
            residuals.abs().mean()
        ),
        "observations": int(len(residuals)),
    }

def build_monitoring_signals(
    residuals: pd.Series,
    window: int = 24,
) -> pd.DataFrame:
    residuals = residuals.dropna().sort_index()

    monitoring = pd.DataFrame(index=residuals.index)

    monitoring["residual"] = residuals

    monitoring["rolling_bias"] = (
        residuals
        .rolling(window=window, min_periods=window)
        .mean()
    )

    monitoring["rolling_mae"] = (
        residuals.abs()
        .rolling(window=window, min_periods=window)
        .mean()
    )

    return monitoring.dropna()