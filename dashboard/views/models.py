import streamlit as st

from components.kpi import render_kpi
from services.model_service import (
    get_production_model,
    get_model_metrics,
    get_model_comparison,
)


def render_models():

    model = get_production_model()
    metrics = get_model_metrics()
    comparison = get_model_comparison()

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Model Registry</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            Production model, validation metrics and candidate comparison
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------------
    # Production model
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Production Model</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 2])

    with col1:

        st.html(
            f"""
            <div class="model-card">
                <div class="model-label">
                    ACTIVE MODEL
                </div>

                <div class="model-name">
                    {model["name"]}
                </div>

                <div class="model-meta">
                    Run {model["version"]}
                </div>

                <div class="model-meta">
                    Status: {model["status"]}
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

    with col2:

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            render_kpi(
                "HOLDOUT MAE",
                (
                    f'{metrics["holdout_mae"]:.2f}'
                    if metrics["holdout_mae"] is not None
                    else "—"
                ),
                "Production validation",
            )

        with metric_col2:
            render_kpi(
                "HOLDOUT RMSE",
                (
                    f'{metrics["holdout_rmse"]:.2f}'
                    if metrics["holdout_rmse"] is not None
                    else "—"
                ),
                "Production validation",
            )

        st.markdown("")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            render_kpi(
                "CV MAE",
                (
                    f'{metrics["cv_mae"]:.2f}'
                    if metrics["cv_mae"] is not None
                    else "—"
                ),
                "Cross-validation",
            )

        with metric_col2:
            render_kpi(
                "CV RMSE",
                (
                    f'{metrics["cv_rmse"]:.2f}'
                    if metrics["cv_rmse"] is not None
                    else "—"
                ),
                "Cross-validation",
            )

    st.markdown("")

    # --------------------------------------------------------------
    # Candidate comparison
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    if comparison.empty:

        st.info(
            "No model comparison report is currently available."
        )

    else:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Run details
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Production Run Details</div>',
        unsafe_allow_html=True,
    )

    st.html(
        f"""
        <div class="model-card">

            <div class="model-label">
                MLFLOW RUN
            </div>

            <div class="model-name">
                {model.get("run_id", "—")}
            </div>

            <div class="model-meta">
                Model: {model["name"]}
            </div>

            <div class="model-meta">
                Variant: {model["variant"]}
            </div>

            <div class="model-meta">
                Data period: {model["data_period"]}
            </div>

            <div class="model-meta">
                Status: {model["status"]}
            </div>

        </div>
        """
    )