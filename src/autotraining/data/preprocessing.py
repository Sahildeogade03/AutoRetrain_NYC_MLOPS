# src/autotraining/data/preprocessing.py

import pandas as pd
import numpy as np


def select_zones(
    hourly_zone_demand: pd.DataFrame,
    coverage_target: float = 0.99,
) -> list[int]:

    zone_totals = (
        hourly_zone_demand
        .groupby("PULocationID")["demand"]
        .sum()
        .sort_values(ascending=False)
    )

    zone_share = zone_totals / zone_totals.sum()
    zone_cumshare = zone_share.cumsum()

    selected_zones = zone_cumshare[
        zone_cumshare <= coverage_target
    ].index.tolist()

    # Include the zone that crosses the threshold.
    if len(selected_zones) < len(zone_cumshare):
        selected_zones.append(
            zone_cumshare.index[len(selected_zones)]
        )

    return selected_zones


def build_zone_panel(
    hourly_zone_demand: pd.DataFrame,
    selected_zones: list[int],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:

    panel = (
        hourly_zone_demand[
            hourly_zone_demand["PULocationID"].isin(selected_zones)
        ]
        .pivot(
            index="timestamp",
            columns="PULocationID",
            values="demand",
        )
        .sort_index()
    )

    expected_hours = pd.date_range(
        start=hourly_zone_demand["timestamp"].min(),
        end=hourly_zone_demand["timestamp"].max(),
        freq="h",
    )

    observed_hours = pd.DatetimeIndex(
        hourly_zone_demand["timestamp"].unique()
    )

    system_missing_hours = expected_hours.difference(
        observed_hours
    )

    panel = panel.reindex(expected_hours)

    panel.index.name = "timestamp"

    # Genuine zero-demand zone/hour combinations.
    observed_system_hours = panel.index.difference(
        system_missing_hours
    )

    panel.loc[observed_system_hours] = (
        panel.loc[observed_system_hours]
        .fillna(0.0)
    )

    # Only interpolate genuine system-wide gaps.
    if len(system_missing_hours) > 0:
        panel = panel.interpolate(
            method="linear",
            limit_direction="both",
        )

    return panel, system_missing_hours