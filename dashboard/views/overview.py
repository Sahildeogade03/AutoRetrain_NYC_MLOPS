import streamlit as st

from components.kpi import render_kpi
from components.charts import actual_vs_forecast_chart
from components.lifecycle import render_lifecycle
from components.monitoring import render_monitoring_summary

from services.forecast_service import get_forecast_metrics
from services.monitoring_service import evaluate_monitoring
from services.model_service import get_production_model
from services.mlflow_service import get_latest_production_run


def render_overview():

    # =========================================================
    # Load backend state
    # =========================================================

    forecast = get_forecast_metrics()
    monitoring = evaluate_monitoring()
    model = get_production_model()
    mlflow_model = get_latest_production_run()

    # =========================================================
    # Production Model KPIs
    # =========================================================

    st.markdown(
        '<div class="section-heading">'
        'Production Model'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    # ---------------------------------------------------------
    # MAE
    # ---------------------------------------------------------

    production_mae = None

    if mlflow_model["available"]:
        production_mae = mlflow_model.get(
            "holdout_mae"
        )

    if production_mae is None:
        production_mae = forecast["mae"]

    with col1:
        render_kpi(
            "MAE",
            f"{production_mae:.2f}",
            "Validated holdout",
        )

    # ---------------------------------------------------------
    # RMSE
    # ---------------------------------------------------------

    production_rmse = None

    if mlflow_model["available"]:
        production_rmse = mlflow_model.get(
            "holdout_rmse"
        )

    if production_rmse is None:
        production_rmse = forecast["rmse"]

    with col2:
        render_kpi(
            "RMSE",
            f"{production_rmse:.2f}",
            "Validated holdout",
        )

    # ---------------------------------------------------------
    # Drift
    # ---------------------------------------------------------

    with col3:

        if monitoring["drift_detected"]:
            render_kpi(
                "DRIFT",
                "DETECTED",
                (
                    f"{len(monitoring['change_points'])} "
                    "change point(s)"
                ),
            )

        else:
            render_kpi(
                "DRIFT",
                "CLEAR",
                "No significant drift",
            )

    # =========================================================
    # Forecast Chart
    # =========================================================

    st.markdown("")

    st.markdown(
        '<div class="section-heading">'
        'Actual vs Forecast'
        '</div>',
        unsafe_allow_html=True,
    )

    fig = actual_vs_forecast_chart(
        forecast["forecast_data"],
        lookback_hours=24 * 7,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    # =========================================================
    # Monitoring Summary
    # =========================================================

    st.markdown("")

    st.markdown(
        '<div class="section-heading">'
        'Monitoring Status'
        '</div>',
        unsafe_allow_html=True,
    )

    render_monitoring_summary(monitoring)

    # =========================================================
    # Model Lifecycle
    # =========================================================

    st.markdown("")

    st.markdown(
        '<div class="section-heading">'
        'Model Lifecycle'
        '</div>',
        unsafe_allow_html=True,
    )

    # The dashboard reflects the monitoring state.
    # It does not execute retraining.

    if monitoring["retrain_required"]:
        lifecycle_stage = "Drift Detected"

    elif monitoring["drift_detected"]:
        lifecycle_stage = "Drift Detected"

    else:
        lifecycle_stage = "Production"

    render_lifecycle(
        active_stage=lifecycle_stage
    )

    # =========================================================
    # Production Model + System Summary
    # =========================================================

    st.markdown("")

    model_col, summary_col = st.columns(
        [1, 2]
    )

    # ---------------------------------------------------------
    # Production Model
    # ---------------------------------------------------------

    with model_col:

        st.html(
            f"""
            <div class="model-card">
                <div class="model-label">
                    PRODUCTION MODEL
                </div>

                <div class="model-name">
                    {model["name"]}
                </div>

                <div class="model-meta">
                    Run {model["version"]} · {model["status"]}
                </div>

                <div class="model-meta">
                    Variant: {model["variant"]}
                </div>

                <div class="model-meta">
                    Data: {model["data_period"]}
                </div>
            </div>
            """
        )

    # ---------------------------------------------------------
    # Determine System Summary
    # ---------------------------------------------------------

    if monitoring["retrain_required"]:

        summary_title = "Retraining required"

    elif monitoring["drift_detected"]:

        summary_title = "Monitoring alert"

    else:

        summary_title = (
            "Forecasting pipeline operational"
        )

    # ---------------------------------------------------------
    # System Summary
    # ---------------------------------------------------------

    with summary_col:

        st.html(
            f"""
            <div class="model-card">
                <div class="model-label">
                    SYSTEM SUMMARY
                </div>

                <div class="model-name">
                    {summary_title}
                </div>

                <div class="model-meta">
                    {monitoring["reason"]}
                </div>

                <div class="model-meta">
                    Latest monitoring data:
                    {forecast["latest_timestamp"]}
                </div>
            </div>
            """
        )