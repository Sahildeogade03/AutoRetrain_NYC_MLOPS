# src/autotraining/data/ingestion.py

from pathlib import Path
import gc

import pandas as pd


REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID",
]


def aggregate_month(
    file_path: Path,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate Yellow Taxi trips into hourly pickup demand by zone.
    """

    df = pd.read_parquet(
        file_path,
        columns=REQUIRED_COLUMNS,
    )

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    df = df[
        (df["tpep_pickup_datetime"] >= month_start)
        & (df["tpep_pickup_datetime"] < month_end)
    ]

    hourly_zone = (
        df.assign(
            timestamp=df["tpep_pickup_datetime"].dt.floor("h")
        )
        .groupby(
            ["timestamp", "PULocationID"],
            observed=True,
        )
        .size()
        .rename("demand")
        .reset_index()
    )

    del df
    gc.collect()

    return hourly_zone


def build_hourly_zone_demand(
    raw_dir: Path,
    months: list[str],
) -> pd.DataFrame:

    monthly_demand = []

    for month in months:

        file_path = raw_dir / f"yellow_tripdata_{month}.parquet"

        month_start = pd.Timestamp(f"{month}-01")
        month_end = month_start + pd.offsets.MonthBegin(1)

        print(f"Processing {month}...")

        month_data = aggregate_month(
            file_path=file_path,
            month_start=month_start,
            month_end=month_end,
        )

        monthly_demand.append(month_data)

        del month_data
        gc.collect()

    return pd.concat(
        monthly_demand,
        ignore_index=True,
    )