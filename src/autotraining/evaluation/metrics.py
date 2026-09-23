# src/autotraining/evaluation/metrics.py

import numpy as np

from darts.metrics import mae, rmse, smape


def compute_metrics(actual, predicted) -> dict:

    actual_values = actual.values(copy=False).squeeze()
    predicted_values = predicted.values(copy=False).squeeze()

    nonzero_mask = actual_values != 0

    if nonzero_mask.any():

        mape_value = (
            np.mean(
                np.abs(
                    (
                        actual_values[nonzero_mask]
                        - predicted_values[nonzero_mask]
                    )
                    / actual_values[nonzero_mask]
                )
            )
            * 100
        )

    else:
        mape_value = np.nan

    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape_value,
        "smape": smape(actual, predicted),
    }


def clip_nonnegative(series_list):

    return [
        series.map(lambda x: np.maximum(x, 0.0))
        for series in series_list
    ]