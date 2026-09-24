from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from darts import TimeSeries

from autotraining.features.demand_features import (
    build_calendar_covariates,
)
from autotraining.models.lightgbm_model import (
    build_lightgbm_raw,
)


CONFIG_PATH = Path("configs/config.yaml")

PANEL_PATH = Path(
    "data/processed/zone_panel.parquet"
)

CANDIDATE_MODEL_PATH = Path(
    "models/lightgbm_raw_candidate.pt"
)

PRODUCTION_MODEL_PATH = Path(
    "models/lightgbm_raw.pt"
)


def load_config(
    config_path: Path = CONFIG_PATH,
) -> dict:
    """Load project configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open("r") as file:
        return yaml.safe_load(file)


def load_zone_panel(
    panel_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    """Load the processed zone demand panel."""

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Zone panel not found: {panel_path}"
        )

    panel = pd.read_parquet(panel_path)

    panel.index = pd.to_datetime(panel.index)

    return panel.sort_index()


def build_zone_series(
    panel: pd.DataFrame,
) -> list[TimeSeries]:
    """Convert each zone into a Darts TimeSeries."""

    return [
        TimeSeries.from_series(panel[zone])
        for zone in panel.columns
    ]


def train_model(
    panel: pd.DataFrame,
    horizon: int,
    device: str = "cpu",
):
    """
    Train LightGBM on the supplied panel.
    """

    zone_series = build_zone_series(panel)

    calendar_covariates = build_calendar_covariates(
        panel.index
    )

    future_covariates = [
        calendar_covariates
        for _ in zone_series
    ]

    model = build_lightgbm_raw(
        horizon=horizon,
        device=device,
    )

    model.fit(
        series=zone_series,
        future_covariates=future_covariates,
    )

    return model


def train_candidate(
    train_panel: pd.DataFrame,
    horizon: int,
    device: str = "cpu",
):
    """
    Train a candidate model using only the training period.
    """

    return train_model(
        panel=train_panel,
        horizon=horizon,
        device=device,
    )


def train_production_model(
    panel: pd.DataFrame,
    horizon: int,
    device: str = "cpu",
):
    """
    Train the final production model using all available data.
    """

    return train_model(
        panel=panel,
        horizon=horizon,
        device=device,
    )


def save_model(
    model,
    output_path: Path,
) -> None:
    """Save a Darts model."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        str(output_path)
    )


def main() -> None:

    print("AutoRetrain-NYC | Production Training")
    print("=" * 60)

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
            336,
        )
    )

    panel = load_zone_panel()

    print(
        f"Training device: {device}"
    )

    print(
        f"Forecast horizon: {horizon}"
    )

    print(
        f"Panel shape: {panel.shape}"
    )

    print(
        f"Training period: "
        f"{panel.index.min()} → "
        f"{panel.index.max()}"
    )

    print("\nTraining production model...")

    model = train_production_model(
        panel=panel,
        horizon=horizon,
        device=device,
    )

    save_model(
        model,
        PRODUCTION_MODEL_PATH,
    )

    print(
        f"\nProduction model saved to: "
        f"{PRODUCTION_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()