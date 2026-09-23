# src/autotraining/models/registry.py

from .baselines import build_naive_model
from .lightgbm_model import (
    build_lightgbm_raw,
    build_lightgbm_residual,
)


MODEL_REGISTRY = {

    "naive_seasonal_24h": {
        "kind": "naive",
        "tier": "Baseline",
        "builder": build_naive_model,
    },

    "lightgbm_raw": {
        "kind": "global_raw",
        "tier": "LightGBM on raw demand",
        "builder": build_lightgbm_raw,
    },

    "lightgbm_residual": {
        "kind": "residual",
        "tier": "LightGBM on naive residuals",
        "builder": build_lightgbm_residual,
    },
}