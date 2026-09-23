# src/autotraining/models/registry.py

from .baselines import build_naive_model
from .lightgbm_model import build_lightgbm_raw


MODEL_REGISTRY = {
    "naive_seasonal_24h": {
        "kind": "naive",
        "tier": "Baseline",
        "builder": build_naive_model,
    },

    "lightgbm_raw": {
        "kind": "global_raw",
        "tier": "Production Candidate",
        "builder": build_lightgbm_raw,
    },
}