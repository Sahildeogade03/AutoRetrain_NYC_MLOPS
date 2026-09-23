# src/autotraining/features/demand_features.py

import pandas as pd
from darts import TimeSeries


CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "month",
]


def build_calendar_covariates(
    index: pd.DatetimeIndex,
) -> TimeSeries:

    calendar_df = pd.DataFrame(index=index)

    calendar_df["hour"] = index.hour
    calendar_df["day_of_week"] = index.dayofweek
    calendar_df["is_weekend"] = (
        calendar_df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )
    calendar_df["day_of_month"] = index.day
    calendar_df["month"] = index.month

    return TimeSeries.from_times_and_values(
        times=calendar_df.index,
        values=calendar_df.values,
        columns=CALENDAR_COLUMNS,
        freq="h",
    )


def build_residual_panel(
    panel: pd.DataFrame,
    seasonal_k: int = 24,
) -> pd.DataFrame:

    naive_panel = panel.shift(seasonal_k)

    return (
        panel - naive_panel
    ).iloc[seasonal_k:]