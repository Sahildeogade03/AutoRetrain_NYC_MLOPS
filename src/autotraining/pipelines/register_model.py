from pathlib import Path

import mlflow
from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_DIR / "models" / "lightgbm_raw_v2.pt"
MLFLOW_DB = PROJECT_DIR / "mlruns.db"

EXPERIMENT_NAME = "AutoRetrain-NYC"
MODEL_NAME = "AutoRetrain-NYC-LightGBM"


def main():

    print("=" * 70)
    print("AutoRetrain-NYC | MLflow Model Registration")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="initial-production-model"
    ) as run:

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "model_type": "LightGBM",
            "model_variant": "raw",
            "seasonal_period": 24,
            "output_horizon": 168,
            "random_state": 42,
            "training_status": "validated_existing_artifact",
        })

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        mlflow.log_metrics({
            "cv_mae": 14.313022,
            "cv_rmse": 18.786759,
            "cv_mape": 139.923026,
            "cv_smape": 86.427560,
            "holdout_mae": 11.378989,
            "holdout_rmse": 15.653094,
            "holdout_mape": 78.824756,
            "holdout_smape": 87.363230,
        })

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tags({
            "project": "AutoRetrain-NYC",
            "model_role": "production",
            "model_variant": "lightgbm_raw",
            "source": "validated_existing_artifact",
            "data_period": "2026-01_to_2026-07",
            "status": "production",
        })

        # ----------------------------------------------------
        # Log actual Darts artifact
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="darts_model",
        )

        # ----------------------------------------------------
        # Create MLflow 3 external model
        # ----------------------------------------------------

        model = mlflow.create_external_model(
            name=MODEL_NAME,
            tags={
                "project": "AutoRetrain-NYC",
                "model_type": "Darts-LightGBM",
                "model_variant": "raw",
                "artifact_path": "darts_model/lightgbm_raw_v2.pt",
            },
        )

        print(
            f"\nMLflow external model created:"
            f"\n{model.model_id}"
        )

        print(
            f"\nRun ID:"
            f"\n{run.info.run_id}"
        )

        print("\nMLflow registration complete.")


if __name__ == "__main__":
    main()