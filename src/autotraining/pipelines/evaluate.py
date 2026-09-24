from pathlib import Path
import json

import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse, smape

from autotraining.features.demand_features import (
    build_calendar_covariates,
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

HOLDOUT_HOURS = 336

# Initial production quality threshold.
# We will make this configurable later.
MAX_MAE_DEGRADATION = 0.05


def compute_mape(actual, predicted):
    actual_values = actual.values(copy=False).squeeze()
    predicted_values = predicted.values(copy=False).squeeze()

    mask = actual_values != 0

    if not mask.any():
        return float("nan")

    return float(
        (
            abs(
                (actual_values[mask] - predicted_values[mask])
                / actual_values[mask]
            ).mean()
            * 100
        )
    )


def main():

    print("=" * 70)
    print("AutoRetrain-NYC | Model Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    panel = pd.read_parquet(
        PROCESSED_DIR / "zone_panel.parquet"
    )

    print(f"\nPanel shape: {panel.shape}")

    # --------------------------------------------------------
    # 2. BUILD DARTS SERIES
    # --------------------------------------------------------

    raw_series = [
        TimeSeries.from_series(panel[zone])
        for zone in panel.columns
    ]

    # --------------------------------------------------------
    # 3. HOLDOUT SPLIT
    # --------------------------------------------------------

    train_series = [
        series[:-HOLDOUT_HOURS]
        for series in raw_series
    ]

    test_series = [
        series[-HOLDOUT_HOURS:]
        for series in raw_series
    ]

    test_index = panel.index[-HOLDOUT_HOURS:]

    future_covariates = build_calendar_covariates(
        panel.index
    )

    train_covariates = future_covariates

    # --------------------------------------------------------
    # 4. LOAD CANDIDATE MODEL
    # --------------------------------------------------------

    candidate_path = (
        MODELS_DIR / "lightgbm_raw.pt"
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {candidate_path}"
        )

    from darts.models import LightGBMModel

    candidate_model = LightGBMModel.load(
        str(candidate_path)
    )

    print("\nCandidate model loaded.")

    # --------------------------------------------------------
    # 5. PREDICT
    # --------------------------------------------------------

    predictions = candidate_model.predict(
        n=HOLDOUT_HOURS,
        series=train_series,
        future_covariates=train_covariates,
    )

    # --------------------------------------------------------
    # 6. CALCULATE METRICS
    # --------------------------------------------------------

    zone_results = []

    for zone, actual, prediction in zip(
        panel.columns,
        test_series,
        predictions,
    ):

        zone_results.append(
            {
                "zone": int(zone),
                "mae": float(
                    mae(actual, prediction)
                ),
                "rmse": float(
                    rmse(actual, prediction)
                ),
                "smape": float(
                    smape(actual, prediction)
                ),
                "mape": compute_mape(
                    actual,
                    prediction,
                ),
            }
        )

    results = pd.DataFrame(zone_results)

    aggregate_metrics = {
        "mae": float(results["mae"].mean()),
        "rmse": float(results["rmse"].mean()),
        "smape": float(results["smape"].mean()),
        "mape": float(results["mape"].mean()),
    }

    # --------------------------------------------------------
    # 7. QUALITY GATE
    # --------------------------------------------------------

    # Existing validated holdout MAE from your model
    # experiment is used as the incumbent reference.
    incumbent_mae = 11.378989

    candidate_mae = aggregate_metrics["mae"]

    allowed_mae = (
        incumbent_mae
        * (1 + MAX_MAE_DEGRADATION)
    )

    passed = candidate_mae <= allowed_mae

    print("\n" + "=" * 70)
    print("QUALITY GATE")
    print("=" * 70)

    print(f"Incumbent MAE : {incumbent_mae:.4f}")
    print(f"Candidate MAE : {candidate_mae:.4f}")
    print(f"Allowed MAE   : {allowed_mae:.4f}")
    print(f"Gate passed   : {passed}")

    # --------------------------------------------------------
    # 8. SAVE METRICS
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "candidate_model": "lightgbm_raw",
        "holdout_hours": HOLDOUT_HOURS,
        "holdout_start": str(test_index[0]),
        "holdout_end": str(test_index[-1]),
        "incumbent_mae": incumbent_mae,
        "candidate_mae": candidate_mae,
        "allowed_mae": allowed_mae,
        "max_mae_degradation": MAX_MAE_DEGRADATION,
        "passed": passed,
        **aggregate_metrics,
    }

    with open(
        REPORTS_DIR / "model_evaluation.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    results.to_csv(
        REPORTS_DIR / "zone_evaluation.csv",
        index=False,
    )

    if not passed:
        raise RuntimeError(
            "Quality gate failed. "
            "Candidate model will not be promoted."
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()