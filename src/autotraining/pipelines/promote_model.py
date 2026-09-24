from pathlib import Path

import mlflow
from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parents[3]
MLFLOW_DB = PROJECT_DIR / "mlruns.db"

MODEL_NAME = "AutoRetrain-NYC-LightGBM"
PRODUCTION_ALIAS = "production"


def main():

    print("=" * 70)
    print("AutoRetrain-NYC | Model Promotion")
    print("=" * 70)

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    client = MlflowClient()

    # Find the most recent registered model version
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            f"No versions found for {MODEL_NAME}"
        )

    latest = max(
        versions,
        key=lambda version: int(version.version),
    )

    print(f"\nModel: {MODEL_NAME}")
    print(f"Version: {latest.version}")

    # Promote latest accepted version
    client.set_registered_model_alias(
        MODEL_NAME,
        PRODUCTION_ALIAS,
        latest.version,
    )

    print(
        f"\nAlias '{PRODUCTION_ALIAS}' "
        f"→ version {latest.version}"
    )

    print("\nPromotion complete.")


if __name__ == "__main__":
    main()