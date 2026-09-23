# src/autotraining/models/lightgbm_model.py

from darts.models import LightGBMModel


LAGS = [
    -1,
    -2,
    -3,
    -24,
    -48,
    -72,
    -96,
    -120,
    -144,
    -168,
]


def build_lightgbm_raw(horizon: int) -> LightGBMModel:

    return LightGBMModel(
        lags=LAGS,
        lags_future_covariates=[0],
        output_chunk_length=horizon,
        random_state=42,
        device="gpu",
        verbose=-1,
    )


def build_lightgbm_residual(
    horizon: int,
) -> LightGBMModel:

    return LightGBMModel(
        lags=None,
        lags_past_covariates=LAGS,
        lags_future_covariates=[0],
        output_chunk_length=horizon,
        random_state=42,
        device="gpu",
        verbose=-1,
    )