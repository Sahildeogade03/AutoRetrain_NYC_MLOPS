from __future__ import annotations

from pathlib import Path

import pandas as pd

from autotraining.evaluation.candidate import (
    evaluate_candidate as evaluate_candidate_model,
)
from autotraining.evaluation.quality_gate import (
    evaluate_quality_gate,
)
from autotraining.monitoring.performance import (
    build_monitoring_signals,
    detect_change_points,
)
from autotraining.pipelines.train import (
    load_config,
    load_zone_panel,
    save_model,
    train_candidate,
    train_production_model,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MONITORING_INPUT = Path(
    "reports/historical_monitoring_forecast.parquet"
)

CANDIDATE_MODEL_PATH = Path(
    "models/lightgbm_raw_candidate.pt"
)

PRODUCTION_MODEL_PATH = Path(
    "models/lightgbm_raw.pt"
)

CANDIDATE_METRICS_PATH = Path(
    "reports/candidate_evaluation.csv"
)


# ---------------------------------------------------------------------
# Monitoring configuration
# ---------------------------------------------------------------------

ROLLING_WINDOW = 24
CPD_MIN_SIZE = 48
CPD_PENALTY = 10.0

PERSISTENCE_WINDOW = 48
MAE_DEGRADATION_THRESHOLD = 0.05

MAX_DEGRADATION = 0.05

HOLDOUT_HORIZON = 336

# Current validated production incumbent.
INCUMBENT_MAE = 11.378989


# ---------------------------------------------------------------------
# Monitoring data
# ---------------------------------------------------------------------

def load_monitoring_data(
    input_path: Path = MONITORING_INPUT,
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Monitoring input not found: {input_path}"
        )

    df = pd.read_parquet(
        input_path
    )

    required_columns = {
        "timestamp",
        "actual_demand",
        "forecast_demand",
        "PULocationID",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing monitoring columns: "
            f"{sorted(missing)}"
        )

    return df


def build_system_residuals(
    df: pd.DataFrame,
) -> pd.Series:

    hourly = (
        df.groupby("timestamp")
        .agg(
            actual_demand=(
                "actual_demand",
                "sum",
            ),
            forecast_demand=(
                "forecast_demand",
                "sum",
            ),
        )
        .sort_index()
    )

    residuals = (
        hourly["actual_demand"]
        - hourly["forecast_demand"]
    )

    residuals.name = "residual"

    return residuals


# ---------------------------------------------------------------------
# Monitoring decision
# ---------------------------------------------------------------------

def evaluate_monitoring(
    residuals: pd.Series,
) -> dict:

    monitoring = build_monitoring_signals(
        residuals=residuals,
        window=ROLLING_WINDOW,
    )

    if monitoring.empty:
        return {
            "drift_detected": False,
            "retrain_required": False,
            "reason": (
                "Insufficient observations."
            ),
        }

    bias_cps = detect_change_points(
        monitoring["rolling_bias"],
        penalty=CPD_PENALTY,
        min_size=CPD_MIN_SIZE,
    )

    mae_cps = detect_change_points(
        monitoring["rolling_mae"],
        penalty=CPD_PENALTY,
        min_size=CPD_MIN_SIZE,
    )

    change_points = sorted(
        set(
            bias_cps
            + mae_cps
        )
    )

    drift_detected = bool(
        change_points
    )

    latest_cp = (
        change_points[-1]
        if change_points
        else None
    )

    latest_change_recent = False

    if latest_cp is not None:

        observations_since_change = (
            len(monitoring)
            - latest_cp
        )

        latest_change_recent = (
            observations_since_change
            <= PERSISTENCE_WINDOW
        )

    historical_mae = float(
        monitoring["rolling_mae"].mean()
    )

    recent_mae = float(
        monitoring
        .tail(PERSISTENCE_WINDOW)[
            "rolling_mae"
        ]
        .mean()
    )

    mae_degradation = (
        (recent_mae - historical_mae)
        / historical_mae
        if historical_mae > 0
        else 0.0
    )

    meaningful_degradation = (
        mae_degradation
        >= MAE_DEGRADATION_THRESHOLD
    )

    retrain_required = (
        drift_detected
        and latest_change_recent
        and meaningful_degradation
    )

    if not drift_detected:

        reason = (
            "No change point detected."
        )

    elif not latest_change_recent:

        reason = (
            "Change point is not recent enough."
        )

    elif not meaningful_degradation:

        reason = (
            "Recent MAE has not degraded "
            "meaningfully."
        )

    else:

        reason = (
            "Persistent drift with meaningful "
            "MAE degradation detected."
        )

    return {
        "drift_detected": drift_detected,
        "retrain_required": retrain_required,
        "reason": reason,
        "bias_change_points": bias_cps,
        "mae_change_points": mae_cps,
        "latest_change_point": latest_cp,
        "historical_mae": historical_mae,
        "recent_mae": recent_mae,
        "mae_degradation": mae_degradation,
        "latest_change_recent": latest_change_recent,
        "meaningful_mae_degradation": (
            meaningful_degradation
        ),
    }


# ---------------------------------------------------------------------
# Candidate training + evaluation
# ---------------------------------------------------------------------

def train_and_evaluate_candidate(
    panel: pd.DataFrame,
    horizon: int,
    device: str,
) -> dict:

    if len(panel) <= horizon:
        raise ValueError(
            "Not enough observations for "
            "candidate holdout."
        )

    train_panel = panel.iloc[
        :-horizon
    ]

    print(
        "\nCandidate training period:"
    )

    print(
        f"{train_panel.index.min()} "
        f"→ "
        f"{train_panel.index.max()}"
    )

    print(
        "\nCandidate holdout period:"
    )

    print(
        f"{panel.index[-horizon]} "
        f"→ "
        f"{panel.index[-1]}"
    )

    print(
        "\nTraining candidate model..."
    )

    candidate_model = train_candidate(
        train_panel=train_panel,
        horizon=horizon,
        device=device,
    )

    save_model(
        candidate_model,
        CANDIDATE_MODEL_PATH,
    )

    print(
        f"\nCandidate saved to: "
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        "\nEvaluating candidate..."
    )

    evaluation = evaluate_candidate_model(
        model=candidate_model,
        panel=panel,
        holdout_horizon=horizon,
    )

    return evaluation


# ---------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------

def evaluate_candidate_quality(
    candidate_mae: float,
    incumbent_mae: float = INCUMBENT_MAE,
) -> dict:

    result = evaluate_quality_gate(
        candidate_metric=candidate_mae,
        incumbent_metric=incumbent_mae,
        max_degradation=MAX_DEGRADATION,
    )

    result["action"] = (
        "promote"
        if result["passed"]
        else "reject"
    )

    return result


# ---------------------------------------------------------------------
# Final production promotion
# ---------------------------------------------------------------------

def promote_candidate(
    panel: pd.DataFrame,
    horizon: int,
    device: str,
) -> Path:

    print(
        "\nQuality gate passed."
    )

    print(
        "Training final production model "
        "on all available data..."
    )

    production_model = train_production_model(
        panel=panel,
        horizon=horizon,
        device=device,
    )

    save_model(
        production_model,
        PRODUCTION_MODEL_PATH,
    )

    print(
        f"\nProduction model saved to: "
        f"{PRODUCTION_MODEL_PATH}"
    )

    return PRODUCTION_MODEL_PATH


# ---------------------------------------------------------------------
# Complete workflow
# ---------------------------------------------------------------------

def run_retraining_workflow() -> dict:

    config = load_config()

    model_config = config.get(
        "model",
        {},
    )

    device = model_config.get(
        "device",
        "cpu",
    )

    horizon = int(
        model_config.get(
            "horizon",
            HOLDOUT_HORIZON,
        )
    )

    # -------------------------------------------------------------
    # 1. Monitoring
    # -------------------------------------------------------------

    monitoring_df = load_monitoring_data()

    residuals = build_system_residuals(
        monitoring_df
    )

    monitoring_result = evaluate_monitoring(
        residuals
    )

    if not monitoring_result[
        "retrain_required"
    ]:

        return {
            "status": "no_retraining",
            "monitoring": monitoring_result,
            "candidate": None,
            "quality_gate": None,
        }

    # -------------------------------------------------------------
    # 2. Load current panel
    # -------------------------------------------------------------

    panel = load_zone_panel()

    # -------------------------------------------------------------
    # 3. Train + evaluate candidate
    # -------------------------------------------------------------

    candidate_result = (
        train_and_evaluate_candidate(
            panel=panel,
            horizon=horizon,
            device=device,
        )
    )

    candidate_mae = candidate_result[
        "mae"
    ]

    # -------------------------------------------------------------
    # 4. Quality gate
    # -------------------------------------------------------------

    quality_result = (
        evaluate_candidate_quality(
            candidate_mae=candidate_mae,
            incumbent_mae=INCUMBENT_MAE,
        )
    )

    # Save candidate evaluation.
    pd.DataFrame(
        [
            {
                "candidate_mae": candidate_mae,
                "candidate_rmse": candidate_result[
                    "rmse"
                ],
                "incumbent_mae": INCUMBENT_MAE,
                "allowed_mae": quality_result[
                    "allowed_metric"
                ],
                "passed": quality_result[
                    "passed"
                ],
                "action": quality_result[
                    "action"
                ],
                "holdout_start": candidate_result[
                    "holdout_start"
                ],
                "holdout_end": candidate_result[
                    "holdout_end"
                ],
            }
        ]
    ).to_csv(
        CANDIDATE_METRICS_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # 5. Reject candidate
    # -------------------------------------------------------------

    if not quality_result["passed"]:

        print(
            "\nQuality gate FAILED."
        )

        print(
            "Keeping incumbent production model."
        )

        return {
            "status": "candidate_rejected",
            "monitoring": monitoring_result,
            "candidate": candidate_result,
            "quality_gate": quality_result,
        }

    # -------------------------------------------------------------
    # 6. Promote candidate
    # -------------------------------------------------------------

    production_path = promote_candidate(
        panel=panel,
        horizon=horizon,
        device=device,
    )

    return {
        "status": "candidate_promoted",
        "monitoring": monitoring_result,
        "candidate": candidate_result,
        "quality_gate": quality_result,
        "production_model": str(
            production_path
        ),
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:

    print(
        "AutoRetrain-NYC | Retraining Pipeline"
    )

    print("=" * 60)

    result = run_retraining_workflow()

    monitoring = result[
        "monitoring"
    ]

    print(
        f"\nDrift detected: "
        f"{monitoring.get('drift_detected')}"
    )

    print(
        f"Retraining required: "
        f"{monitoring.get('retrain_required')}"
    )

    print(
        f"Reason: "
        f"{monitoring.get('reason')}"
    )

    if result["candidate"] is not None:

        candidate = result[
            "candidate"
        ]

        gate = result[
            "quality_gate"
        ]

        print(
            "\nCandidate MAE: "
            f"{candidate['mae']:.6f}"
        )

        print(
            "Incumbent MAE: "
            f"{gate['incumbent_metric']:.6f}"
        )

        print(
            "Allowed MAE: "
            f"{gate['allowed_metric']:.6f}"
        )

        print(
            "Quality gate: "
            f"{'PASSED' if gate['passed'] else 'FAILED'}"
        )

        print(
            "Action: "
            f"{gate['action'].upper()}"
        )

    print(
        f"\nFinal status: "
        f"{result['status']}"
    )


if __name__ == "__main__":
    main()