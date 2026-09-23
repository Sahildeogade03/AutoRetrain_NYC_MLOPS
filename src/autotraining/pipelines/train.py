# src/autotraining/pipelines/train.py

from pathlib import Path

from autotraining.data.ingestion import (
    build_hourly_zone_demand,
)

from autotraining.data.preprocessing import (
    select_zones,
    build_zone_panel,
)

from autotraining.features.demand_features import (
    build_calendar_covariates,
    build_residual_panel,
)

from autotraining.models.registry import (
    MODEL_REGISTRY,
)


def run_training(config):

    print("=" * 70)
    print("AutoRetrain-NYC Training Pipeline")
    print("=" * 70)

    # --------------------------------------------------
    # 1. DATA INGESTION
    # --------------------------------------------------

    hourly_zone_demand = build_hourly_zone_demand(
        raw_dir=config.raw_dir,
        months=config.months,
    )

    # --------------------------------------------------
    # 2. ZONE SELECTION
    # --------------------------------------------------

    selected_zones = select_zones(
        hourly_zone_demand,
        coverage_target=config.coverage_target,
    )

    # --------------------------------------------------
    # 3. BUILD PANEL
    # --------------------------------------------------

    panel, system_missing_hours = build_zone_panel(
        hourly_zone_demand,
        selected_zones,
    )

    # --------------------------------------------------
    # 4. FEATURES
    # --------------------------------------------------

    residual_panel = build_residual_panel(
        panel,
        seasonal_k=config.seasonal_k,
    )

    # Calendar features
    calendar_covariates = build_calendar_covariates(
        panel.index
    )

    # --------------------------------------------------
    # 5. MODEL TRAINING / CV
    # --------------------------------------------------

    # This will call run_walk_forward_cv(...)
    # after we extract that module.

    # --------------------------------------------------
    # 6. HOLDOUT
    # --------------------------------------------------

    # fit_and_predict_holdout(...)

    # --------------------------------------------------
    # 7. MODEL SELECTION
    # --------------------------------------------------

    # comparison_table

    # --------------------------------------------------
    # 8. FINAL FIT
    # --------------------------------------------------

    # fit selected model on all history

    # --------------------------------------------------
    # 9. SAVE ARTIFACTS
    # --------------------------------------------------

    # model
    # metrics
    # inference