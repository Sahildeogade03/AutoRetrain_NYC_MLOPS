from pathlib import Path

import pandas as pd

from autotraining.monitoring.performance import (
    evaluate_residual_drift,
    build_monitoring_signals,
    detect_change_points,
)


INPUT_PATH = Path(
    "reports/historical_monitoring_forecast.parquet"
)


def main():

    df = pd.read_parquet(INPUT_PATH)

    # ---------------------------------------------------------
    # Aggregate zone-level forecasts and actuals
    # into a system-wide hourly signal.
    # ---------------------------------------------------------

    hourly = (
        df.groupby("timestamp")
        .agg(
            actual_demand=("actual_demand", "sum"),
            forecast_demand=("forecast_demand", "sum"),
        )
        .sort_index()
    )

    hourly["residual"] = (
        hourly["actual_demand"]
        - hourly["forecast_demand"]
    )

    residuals = hourly["residual"]

    print("AutoRetrain-NYC | CPD Calibration")
    print("-" * 50)

    print(f"Observations: {len(residuals)}")
    print(f"Residual mean: {residuals.mean():.3f}")
    print(f"Residual std: {residuals.std():.3f}")
    print(f"System MAE: {residuals.abs().mean():.3f}")

    # =========================================================
    # 1. RAW RESIDUAL CPD
    # =========================================================

    print("\nRaw residual CPD")
    print("-" * 50)

    penalties = [5, 10, 20, 30, 50]

    for penalty in penalties:

        result = evaluate_residual_drift(
            residuals=residuals,
            penalty=penalty,
            min_size=24,
        )

        print(f"\nPenalty = {penalty}")
        print(f"  Drift detected: {result['drift_detected']}")
        print(f"  Bias CPs: {result['bias_change_points']}")
        print(f"  Magnitude CPs: {result['magnitude_change_points']}")

    # =========================================================
    # 2. SMOOTHED MONITORING SIGNALS
    # =========================================================

    print("\n\nSmoothed monitoring signals")
    print("-" * 50)

    monitoring = build_monitoring_signals(
        residuals=residuals,
        window=24,
    )

    print(f"Monitoring observations: {len(monitoring)}")
    print(
        f"Rolling bias mean: "
        f"{monitoring['rolling_bias'].mean():.3f}"
    )
    print(
        f"Rolling MAE mean: "
        f"{monitoring['rolling_mae'].mean():.3f}"
    )

    # =========================================================
    # 3. CPD ON SMOOTHED SIGNALS
    # =========================================================

    print("\nSmoothed CPD penalty comparison")
    print("-" * 50)

    for penalty in penalties:

        bias_cps = detect_change_points(
            monitoring["rolling_bias"],
            penalty=penalty,
            min_size=48,
        )

        mae_cps = detect_change_points(
            monitoring["rolling_mae"],
            penalty=penalty,
            min_size=48,
        )

        print(f"\nPenalty = {penalty}")
        print(f"  Rolling-bias CPs: {bias_cps}")
        print(f"  Rolling-MAE CPs:  {mae_cps}")


if __name__ == "__main__":
    main()