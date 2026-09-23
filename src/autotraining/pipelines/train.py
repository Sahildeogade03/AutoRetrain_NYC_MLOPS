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


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

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


# ============================================================
# DATA PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("AutoRetrain-NYC | Data Pipeline")
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
        f"Selected zones: {len(selected_zones)}"
    )

    # --------------------------------------------------------
    # 3. BUILD PANEL
    # --------------------------------------------------------

    print("\n[3/4] Building hourly zone panel...")

    panel, system_missing_hours = build_zone_panel(
        hourly_zone_demand,
        selected_zones,
    )

    print(f"Panel shape: {panel.shape}")

    print(
        f"System-wide missing hours: "
        f"{len(system_missing_hours)}"
    )

    # --------------------------------------------------------
    # 4. FEATURES
    # --------------------------------------------------------

    print("\n[4/4] Building features...")

    residual_panel = build_residual_panel(
        panel,
        seasonal_k=SEASONAL_K,
    )

    calendar_covariates = build_calendar_covariates(
        panel.index
    )

    print(
        f"Residual panel shape: "
        f"{residual_panel.shape}"
    )

    print(
        f"Calendar covariates: "
        f"{calendar_covariates}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA PIPELINE COMPLETE")
    print("=" * 70)

    print(f"Date range : {panel.index.min()} → {panel.index.max()}")
    print(f"Zones      : {len(selected_zones)}")
    print(f"Panel      : {panel.shape}")
    print(f"Residual   : {residual_panel.shape}")

    # Save intermediate data
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_zone_demand.to_parquet(
        PROCESSED_DIR /
        "hourly_zone_demand_pipeline.parquet",
        index=False,
    )

    panel.to_parquet(
        PROCESSED_DIR /
        "zone_panel_pipeline.parquet",
    )

    residual_panel.to_parquet(
        PROCESSED_DIR /
        "residual_panel_pipeline.parquet",
    )


if __name__ == "__main__":
    main()