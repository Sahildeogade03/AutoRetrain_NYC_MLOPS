# src/autotraining/pipelines/train.py

from pathlib import Path

import pandas as pd
from darts import TimeSeries

from autotraining.data.ingestion import (
    build_hourly_zone_demand,
)

from autotraining.data.preprocessing import (
    select_zones,
    build_zone_panel,
)

from autotraining.features.demand_features import (
    build_calendar_covariates,
)

from autotraining.models.registry import (
    MODEL_REGISTRY,
)

from autotraining.evaluation.cross_validation import (
    make_walk_forward_folds,
    run_walk_forward_cv,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"


# ============================================================
# CONFIG
# ============================================================

MONTHS = [
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
    "2026-07",
]

COVERAGE_TARGET = 0.99
SEASONAL_K = 24

# Walk-forward validation
WF_HORIZON = 48
WF_STRIDE = 7 * 24
WF_MIN_TRAIN = 60 * 24


# ============================================================
# DATA PIPELINE
# ============================================================

def build_data_pipeline():

    print("=" * 70)
    print("AutoRetrain-NYC | Data + Feature Pipeline")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. INGESTION
    # --------------------------------------------------------

    print("\n[1/4] Building hourly zone demand...")

    hourly_zone_demand = build_hourly_zone_demand(
        raw_dir=RAW_DIR,
        months=MONTHS,
    )

    print(
        f"Hourly zone demand shape: "
        f"{hourly_zone_demand.shape}"
    )

    # --------------------------------------------------------
    # 2. ZONE SELECTION
    # --------------------------------------------------------

    print("\n[2/4] Selecting demand-covering zones...")

    selected_zones = select_zones(
        hourly_zone_demand,
        coverage_target=COVERAGE_TARGET,
    )

    print(
        f"Selected zones: "
        f"{len(selected_zones)}"
    )

    # --------------------------------------------------------
    # 3. BUILD HOURLY PANEL
    # --------------------------------------------------------

    print("\n[3/4] Building hourly zone panel...")

    panel, system_missing_hours = build_zone_panel(
        hourly_zone_demand,
        selected_zones,
    )

    print(
        f"Panel shape: "
        f"{panel.shape}"
    )

    print(
        f"System-wide missing hours: "
        f"{len(system_missing_hours)}"
    )

    # --------------------------------------------------------
    # 4. FEATURES
    # --------------------------------------------------------

    print("\n[4/4] Building calendar features...")

    calendar_covariates = build_calendar_covariates(
        panel.index
    )

    print(
        f"Calendar covariates shape: "
        f"{calendar_covariates.shape}"
    )

    # --------------------------------------------------------
    # SAVE INTERMEDIATE DATA
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_zone_demand.to_parquet(
        PROCESSED_DIR
        / "hourly_zone_demand_pipeline.parquet",
        index=False,
    )

    panel.to_parquet(
        PROCESSED_DIR
        / "zone_panel_pipeline.parquet",
    )

    print("\nData pipeline complete.")

    return (
        panel,
        selected_zones,
        calendar_covariates,
    )


# ============================================================
# BUILD DARTS SERIES
# ============================================================

def build_darts_series(panel):

    print("\n" + "=" * 70)
    print("Building Darts TimeSeries objects")
    print("=" * 70)

    raw_series = []

    for zone in panel.columns:

        zone_series = TimeSeries.from_series(
            panel[zone]
        )

        raw_series.append(
            zone_series
        )

    print(
        f"Created {len(raw_series)} "
        f"zone time series."
    )

    return raw_series


# ============================================================
# WALK-FORWARD CROSS VALIDATION
# ============================================================

def run_cv(
    raw_series,
    calendar_covariates,
):

    print("\n" + "=" * 70)
    print("Walk-Forward Cross-Validation")
    print("=" * 70)

    total_length = len(raw_series[0])

    folds = make_walk_forward_folds(
        total_length=total_length,
        min_train_size=WF_MIN_TRAIN,
        horizon=WF_HORIZON,
        stride=WF_STRIDE,
    )

    print(
        f"\nTotal observations : {total_length}"
    )

    print(
        f"Minimum train size : {WF_MIN_TRAIN}"
    )

    print(
        f"Validation horizon  : {WF_HORIZON}"
    )

    print(
        f"Validation stride   : {WF_STRIDE}"
    )

    print(
        f"Number of folds     : {len(folds)}"
    )

    all_results = []

    # --------------------------------------------------------
    # RUN EACH MODEL
    # --------------------------------------------------------

    for model_name, model_cfg in MODEL_REGISTRY.items():

        print("\n" + "-" * 70)

        print(
            f"Running model: "
            f"{model_name}"
        )

        print(
            f"Model kind: "
            f"{model_cfg['kind']}"
        )

        print("-" * 70)

        results = run_walk_forward_cv(
            raw_series=raw_series,
            future_covariates=calendar_covariates,
            folds=folds,
            model_name=model_name,
            model_cfg=model_cfg,
            seasonal_k=SEASONAL_K,
        )

        all_results.append(
            results
        )

    cv_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    return cv_results


# ============================================================
# SUMMARIZE CV RESULTS
# ============================================================

def summarize_cv_results(
    cv_results,
):

    print("\n" + "=" * 70)
    print("CV RESULTS")
    print("=" * 70)

    comparison = (
        cv_results
        .groupby("model")
        .agg(
            cv_mae=("mae", "mean"),
            cv_rmse=("rmse", "mean"),
            cv_mape=("mape", "mean"),
            cv_smape=("smape", "mean"),
        )
        .sort_values("cv_mae")
    )

    print("\nModel comparison:\n")

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv_results.to_csv(
        REPORTS_DIR
        / "walk_forward_cv_results.csv",
        index=False,
    )

    comparison.to_csv(
        REPORTS_DIR
        / "cv_model_comparison.csv",
    )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    selection_metric = "cv_mae"

    best_model_name = (
        comparison[selection_metric]
        .idxmin()
    )

    print(
        "\nSelected model by "
        f"{selection_metric}: "
        f"{best_model_name}"
    )

    return comparison, best_model_name


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. DATA + FEATURES
    # --------------------------------------------------------

    (
        panel,
        selected_zones,
        calendar_covariates,
    ) = build_data_pipeline()

    # --------------------------------------------------------
    # 2. DARTS SERIES
    # --------------------------------------------------------

    raw_series = build_darts_series(
        panel
    )

    # --------------------------------------------------------
    # 3. WALK-FORWARD CV
    # --------------------------------------------------------

    cv_results = run_cv(
        raw_series=raw_series,
        calendar_covariates=calendar_covariates,
    )

    # --------------------------------------------------------
    # 4. MODEL COMPARISON
    # --------------------------------------------------------

    comparison, best_model_name = (
        summarize_cv_results(
            cv_results
        )
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Date range : "
        f"{panel.index.min()} → "
        f"{panel.index.max()}"
    )

    print(
        f"Zones      : "
        f"{len(selected_zones)}"
    )

    print(
        f"Panel      : "
        f"{panel.shape}"
    )

    print(
        f"CV folds   : "
        f"{cv_results['fold'].nunique()}"
    )

    print(
        f"Models     : "
        f"{cv_results['model'].nunique()}"
    )

    print(
        f"Selected   : "
        f"{best_model_name}"
    )


if __name__ == "__main__":
    main()