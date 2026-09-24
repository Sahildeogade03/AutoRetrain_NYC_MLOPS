import numpy as np
import pandas as pd

from autotraining.monitoring.performance import (
    calculate_residuals,
    calculate_mae,
    detect_change_points,
    evaluate_residual_drift,
    build_monitoring_signals
)


def test_calculate_residuals():

    actual = pd.Series(
        [10, 20, 30],
        index=pd.date_range("2026-01-01", periods=3, freq="h"),
    )

    forecast = pd.Series(
        [8, 18, 33],
        index=actual.index,
    )

    residuals = calculate_residuals(
        actual,
        forecast,
    )

    assert residuals.tolist() == [2, 2, -3]


def test_calculate_mae():

    actual = pd.Series([10, 20, 30])
    forecast = pd.Series([8, 18, 33])

    mae = calculate_mae(
        actual,
        forecast,
    )

    assert np.isclose(mae, 7 / 3)


def test_change_point_detection():

    rng = np.random.default_rng(42)

    residuals = np.concatenate(
        [
            rng.normal(0, 1, 200),
            rng.normal(10, 1, 200),
        ]
    )

    series = pd.Series(
        residuals,
        index=pd.date_range(
            "2026-01-01",
            periods=len(residuals),
            freq="h",
        ),
    )

    change_points = detect_change_points(
        series,
        penalty=10,
        min_size=24,
    )

    assert len(change_points) >= 1


def test_residual_drift_detection():

    rng = np.random.default_rng(42)

    residuals = np.concatenate(
        [
            rng.normal(0, 1, 200),
            rng.normal(10, 1, 200),
        ]
    )

    series = pd.Series(residuals)

    result = evaluate_residual_drift(
        series,
        penalty=10,
        min_size=24,
    )

    assert result["drift_detected"] is True
    assert result["observations"] == 400


def test_build_monitoring_signals():
    index = pd.date_range(
        "2026-01-01",
        periods=72,
        freq="h",
    )

    residuals = pd.Series(
        range(72),
        index=index,
        dtype=float,
    )

    monitoring = build_monitoring_signals(
        residuals,
        window=24,
    )

    assert len(monitoring) == 49
    assert "residual" in monitoring.columns
    assert "rolling_bias" in monitoring.columns
    assert "rolling_mae" in monitoring.columns

    assert monitoring["rolling_bias"].notna().all()
    assert monitoring["rolling_mae"].notna().all()