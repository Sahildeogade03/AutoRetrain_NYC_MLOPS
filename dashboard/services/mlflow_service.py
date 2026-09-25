from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB = PROJECT_ROOT / "mlruns.db"

EXPERIMENT_NAME = "AutoRetrain-NYC"


def get_mlflow_client() -> MlflowClient:
    """Create an MLflow client using the project's tracking DB."""

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    return MlflowClient()


def get_latest_production_run() -> dict:
    """
    Get the latest MLflow run tagged as the production model.
    """

    client = get_mlflow_client()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        return {
            "available": False,
            "reason": "MLflow experiment not found.",
        }

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.model_role = 'production'"
        ),
        order_by=[
            "start_time DESC"
        ],
        max_results=1,
    )

    if not runs:
        return {
            "available": False,
            "reason": "No production MLflow run found.",
        }

    run = runs[0]

    metrics = run.data.metrics
    tags = run.data.tags

    return {
        "available": True,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,

        "model_name": (
            "AutoRetrain-NYC-LightGBM"
        ),

        "model_variant": tags.get(
            "model_variant",
            "unknown",
        ),

        "model_role": tags.get(
            "model_role",
            "unknown",
        ),

        "status": tags.get(
            "status",
            "unknown",
        ),

        "data_period": tags.get(
            "data_period",
            "unknown",
        ),

        "cv_mae": metrics.get("cv_mae"),
        "cv_rmse": metrics.get("cv_rmse"),

        "holdout_mae": metrics.get(
            "holdout_mae"
        ),

        "holdout_rmse": metrics.get(
            "holdout_rmse"
        ),

        "holdout_mape": metrics.get(
            "holdout_mape"
        ),

        "holdout_smape": metrics.get(
            "holdout_smape"
        ),
    }