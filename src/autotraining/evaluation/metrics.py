from __future__ import annotations

import pandas as pd


def load_holdout_results(
    path: str = "reports/v2_final_holdout_results.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "zone",
        "model",
        "mae",
        "rmse",
        "mape",
        "smape",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def summarize_model_metrics(
    df: pd.DataFrame,
    model_name: str,
) -> dict:
    model_df = df[df["model"] == model_name]

    if model_df.empty:
        raise ValueError(
            f"No holdout results found for model: {model_name}"
        )

    return {
        "model": model_name,
        "mae": float(model_df["mae"].mean()),
        "rmse": float(model_df["rmse"].mean()),
        "mape": float(model_df["mape"].mean()),
        "smape": float(model_df["smape"].mean()),
        "zones": int(model_df["zone"].nunique()),
    }