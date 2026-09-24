from pathlib import Path
import json

import pandas as pd

from autotraining.monitoring.performance import (
    calculate_residuals,
    evaluate_residual_drift,
)


FORECAST_PATH = Path(
    "reports/v2_inference_forecast_next_7d.parquet"
)

ACTUAL_PATH = Path(
    "data/processed/hourly_zone_demand.parquet"
)

OUTPUT_PATH = Path(
    "reports/residual_cpd_report.json"
)


def main():

    print("AutoRetrain-NYC | Residual CPD Monitoring")
    print("-" * 50)

    forecast = pd.read_parquet(
        FORECAST_PATH
    )

    actual = pd.read_parquet(
        ACTUAL_PATH
    )

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"]
    )

    actual["timestamp"] = pd.to_datetime(
        actual["timestamp"]
    )

    forecast_start = forecast["timestamp"].min()
    forecast_end = forecast["timestamp"].max()

    actual_start = actual["timestamp"].min()
    actual_end = actual["timestamp"].max()

    print(
        f"Forecast window: "
        f"{forecast_start} → {forecast_end}"
    )

    print(
        f"Actual window:   "
        f"{actual_start} → {actual_end}"
    )

    # ---------------------------------------------------------
    # Check whether actuals exist for forecast period
    # ---------------------------------------------------------

    evaluation_actuals = actual[
        actual["timestamp"].between(
            forecast_start,
            forecast_end,
        )
    ]

    if evaluation_actuals.empty:

        print(
            "\nNo actual observations are available "
            "for the forecast window."
        )

        print(
            "Residual monitoring will remain pending "
            "until actuals arrive."
        )

        result = {
            "status": "pending_actuals",
            "drift_detected": False,
            "forecast_window_start": str(
                forecast_start
            ),
            "forecast_window_end": str(
                forecast_end
            ),
            "actual_data_available_until": str(
                actual_end
            ),
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        return

    # ---------------------------------------------------------
    # Align forecast and actual demand
    # ---------------------------------------------------------

    merged = forecast.merge(
        actual,
        on=["timestamp", "PULocationID"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "Forecast and actual data overlap in time "
            "but no zone-level observations matched."
        )

    # ---------------------------------------------------------
    # Aggregate system-wide demand
    # ---------------------------------------------------------

    system_forecast = (
        merged
        .groupby("timestamp")["forecast_demand"]
        .sum()
        .sort_index()
    )

    system_actual = (
        merged
        .groupby("timestamp")["demand"]
        .sum()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Residuals
    # ---------------------------------------------------------

    residuals = calculate_residuals(
        actual=system_actual,
        forecast=system_forecast,
    )

    # ---------------------------------------------------------
    # Change-point detection
    # ---------------------------------------------------------

    result = evaluate_residual_drift(
        residuals=residuals,
        penalty=10.0,
        min_size=24,
    )

    result.update(
        {
            "status": "evaluated",
            "forecast_window_start": str(
                forecast_start
            ),
            "forecast_window_end": str(
                forecast_end
            ),
            "actual_data_available_until": str(
                actual_end
            ),
            "cpd_penalty": 10.0,
            "cpd_min_size": 24,
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nResidual CPD Summary")
    print("-" * 50)

    print(
        f"Status: {result['status']}"
    )

    print(
        f"Drift detected: "
        f"{result['drift_detected']}"
    )

    print(
        f"Residual MAE: "
        f"{result['mean_absolute_error']:.4f}"
    )

    print(
        f"Residual mean: "
        f"{result['residual_mean']:.4f}"
    )

    print(
        f"Bias change points: "
        f"{result['bias_change_points']}"
    )

    print(
        f"Magnitude change points: "
        f"{result['magnitude_change_points']}"
    )

    print(
        f"\nReport saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()