# src/autotraining/models/baselines.py

from darts.models import NaiveSeasonal


def build_naive_model(
    seasonal_period: int = 24,
) -> NaiveSeasonal:

    return NaiveSeasonal(K=seasonal_period)