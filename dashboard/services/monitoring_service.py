from pathlib import Path

import pandas as pd

from autotraining.monitoring.performance import (
    build_monitoring_signals,
    detect_change_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONITORING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "historical_monitoring_forecast.parquet"
)

ROLLING_WINDOW = 24
CPD_MIN_SIZE = 48
CPD_PENALTY = 10.0
PERSISTENCE_WINDOW = 48
MAE_DEGRADATION_THRESHOLD = 0.05


def load_monitoring_data() -> pd.DataFrame:
    """Load the persisted zone-level monitoring forecast."""

    if not MONITORING_PATH.exists():
        raise FileNotFoundError(
            f"Monitoring file not found: {MONITORING_PATH}"
        )

    df = pd.read_parquet(MONITORING_PATH)

    required_columns = {
        "timestamp",
        "forecast_demand",
        "actual_demand",
        "residual",
        "PULocationID",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


def get_system_residuals() -> pd.Series:
    """
    Aggregate zone-level residuals into
    system-wide hourly residuals.
    """

    df = load_monitoring_data()

    residuals = (
        df.groupby("timestamp")["residual"]
        .sum()
        .sort_index()
    )

    residuals.name = "residual"

    return residuals


def evaluate_monitoring() -> dict:
    """
    Evaluate current production monitoring state
    using the project's existing monitoring logic.
    """

    residuals = get_system_residuals()

    monitoring = build_monitoring_signals(
        residuals,
        window=ROLLING_WINDOW,
    )

    # ---------------------------------------------------------
    # Change point detection
    # ---------------------------------------------------------

    bias_change_points = detect_change_points(
        monitoring["rolling_bias"],
        min_size=CPD_MIN_SIZE,
        penalty=CPD_PENALTY,
    )

    magnitude_change_points = detect_change_points(
        monitoring["rolling_mae"],
        min_size=CPD_MIN_SIZE,
        penalty=CPD_PENALTY,
    )

    change_points = sorted(
        set(
            bias_change_points
            + magnitude_change_points
        )
    )

    drift_detected = bool(change_points)

    # ---------------------------------------------------------
    # Latest change point
    # ---------------------------------------------------------

    latest_change_point = None

    if change_points:

        latest_cp = change_points[-1]

        position = min(
            max(latest_cp - 1, 0),
            len(monitoring.index) - 1,
        )

        latest_change_point = (
            monitoring.index[position]
        )

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    recent_change = False

    if latest_change_point is not None:

        latest_position = (
            monitoring.index.get_loc(
                latest_change_point
            )
        )

        observations_since_change = (
            len(monitoring)
            - latest_position
            - 1
        )

        recent_change = (
            observations_since_change
            <= PERSISTENCE_WINDOW
        )

    # ---------------------------------------------------------
    # MAE degradation
    # ---------------------------------------------------------

    historical_mae = float(
        monitoring["rolling_mae"].mean()
    )

    recent_mae = float(
        monitoring.tail(
            PERSISTENCE_WINDOW
        )["rolling_mae"].mean()
    )

    if historical_mae > 0:

        mae_degradation = (
            recent_mae - historical_mae
        ) / historical_mae

    else:

        mae_degradation = 0.0

    meaningful_degradation = (
        mae_degradation
        >= MAE_DEGRADATION_THRESHOLD
    )

    # ---------------------------------------------------------
    # Retraining decision
    # ---------------------------------------------------------

    retrain_required = (
        drift_detected
        and recent_change
        and meaningful_degradation
    )

    if retrain_required:

        reason = (
            "Recent distribution change "
            "accompanied by meaningful "
            "MAE degradation."
        )

    elif drift_detected:

        reason = (
            "Change point detected, but "
            "retraining conditions are "
            "not currently satisfied."
        )

    else:

        reason = (
            "No significant drift detected."
        )

    return {
        "drift_detected": drift_detected,
        "retrain_required": retrain_required,
        "reason": reason,

        "change_points": change_points,

        "bias_change_points": (
            bias_change_points
        ),

        "magnitude_change_points": (
            magnitude_change_points
        ),

        "latest_change_point": (
            latest_change_point
        ),

        "historical_mae": historical_mae,
        "recent_mae": recent_mae,

        "mae_degradation": float(
            mae_degradation
        ),

        "recent_change": recent_change,

        "meaningful_degradation": (
            meaningful_degradation
        ),

        "monitoring": monitoring,
    }