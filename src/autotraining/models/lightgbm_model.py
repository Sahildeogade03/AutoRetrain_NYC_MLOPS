from __future__ import annotations

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


def build_lightgbm_raw(
    horizon: int,
    device: str = "cpu",
) -> LightGBMModel:
    """
    Build the production LightGBM model.

    The device is configurable so the same training code
    can run on CPU or GPU environments.
    """

    return LightGBMModel(
        lags=LAGS,
        lags_future_covariates=[0],
        output_chunk_length=horizon,
        random_state=42,
        device=device,
        verbose=-1,
    )


def build_lightgbm_residual(
    horizon: int,
    device: str = "cpu",
) -> LightGBMModel:
    """
    Build the residual LightGBM model used for experiments.
    """

    return LightGBMModel(
        lags=None,
        lags_past_covariates=LAGS,
        lags_future_covariates=[0],
        output_chunk_length=horizon,
        random_state=42,
        device=device,
        verbose=-1,
    )