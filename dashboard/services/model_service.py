from .mlflow_service import get_latest_production_run


def get_production_model() -> dict:
    """Return production model information for the dashboard."""

    model = get_latest_production_run()

    if not model["available"]:
        return {
            "name": "Unavailable",
            "version": "—",
            "status": "Unknown",
            "variant": "—",
            "data_period": "—",
            "run_id": None,
        }

    return {
        "name": model["model_name"],
        "version": model["run_id"][:8],
        "status": model["status"].title(),
        "variant": model["model_variant"],
        "data_period": model["data_period"],
        "run_id": model["run_id"],
    }

def get_model_metrics():
    model = get_latest_production_run()

    if not model["available"]:
        return {
            "available": False,
            "holdout_mae": None,
            "holdout_rmse": None,
            "cv_mae": None,
            "cv_rmse": None,
        }

    return {
        "available": True,
        "holdout_mae": model.get("holdout_mae"),
        "holdout_rmse": model.get("holdout_rmse"),
        "cv_mae": model.get("cv_mae"),
        "cv_rmse": model.get("cv_rmse"),
    }

def get_model_comparison():
    from pathlib import Path
    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]

    comparison_path = (
        project_root
        / "reports"
        / "v2_model_comparison_summary.csv"
    )

    if not comparison_path.exists():
        return pd.DataFrame()

    return pd.read_csv(comparison_path)

def get_experiment_results():
    from pathlib import Path
    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]

    comparison_path = (
        project_root
        / "reports"
        / "v2_model_comparison_summary.csv"
    )

    cv_path = (
        project_root
        / "reports"
        / "v2_walk_forward_cv_results.csv"
    )

    comparison = (
        pd.read_csv(comparison_path)
        if comparison_path.exists()
        else pd.DataFrame()
    )

    walk_forward = (
        pd.read_csv(cv_path)
        if cv_path.exists()
        else pd.DataFrame()
    )

    return {
        "comparison": comparison,
        "walk_forward": walk_forward,
    }