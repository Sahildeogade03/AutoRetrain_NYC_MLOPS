import streamlit as st

from components.kpi import render_kpi
from services.model_service import get_experiment_results


def render_experiments():

    # --------------------------------------------------------------
    # Load experiment artifacts
    # --------------------------------------------------------------

    results = get_experiment_results()

    comparison = results["comparison"]
    walk_forward = results["walk_forward"]

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Experiment Tracking</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            Model validation, candidate comparison and experiment results
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Experiment Summary</div>',
        unsafe_allow_html=True,
    )

    candidate_count = (
        len(comparison)
        if not comparison.empty
        else 0
    )

    cv_count = (
        len(walk_forward)
        if not walk_forward.empty
        else 0
    )

    # Try to identify MAE / RMSE columns without assuming
    # an exact report schema.
    mae_columns = [
        column
        for column in comparison.columns
        if "mae" in column.lower()
    ]

    rmse_columns = [
        column
        for column in comparison.columns
        if "rmse" in column.lower()
    ]

    best_mae = None
    best_rmse = None

    if mae_columns:
        values = comparison[mae_columns[0]].dropna()

        if not values.empty:
            best_mae = values.min()

    if rmse_columns:
        values = comparison[rmse_columns[0]].dropna()

        if not values.empty:
            best_rmse = values.min()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi(
            "CANDIDATES",
            str(candidate_count),
            "Models evaluated",
        )

    with col2:
        render_kpi(
            "CV RUNS",
            str(cv_count),
            "Walk-forward results",
        )

    with col3:
        render_kpi(
            "BEST MAE",
            f"{best_mae:.2f}"
            if best_mae is not None
            else "—",
            "Candidate comparison",
        )

    with col4:
        render_kpi(
            "BEST RMSE",
            f"{best_rmse:.2f}"
            if best_rmse is not None
            else "—",
            "Candidate comparison",
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    if comparison.empty:

        st.info(
            "No model comparison experiment is currently available."
        )

    else:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Walk-forward validation
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Walk-Forward Validation</div>',
        unsafe_allow_html=True,
    )

    if walk_forward.empty:

        st.info(
            "No walk-forward validation results are currently available."
        )

    else:

        st.dataframe(
            walk_forward,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Experiment Configuration</div>',
        unsafe_allow_html=True,
    )

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.html(
            """
            <div class="model-card">

                <div class="model-label">
                    VALIDATION
                </div>

                <div class="model-name">
                    Walk-Forward Validation
                </div>

                <div class="model-meta">
                    Time-series aware evaluation
                </div>

                <div class="model-meta">
                    Validation results are stored in the
                    experiment report.
                </div>

            </div>
            """
        )

    with config_col2:

        st.html(
            """
            <div class="model-card">

                <div class="model-label">
                    MODEL SELECTION
                </div>

                <div class="model-name">
                    Holdout Performance
                </div>

                <div class="model-meta">
                    Candidate models are compared using
                    validation metrics.
                </div>

                <div class="model-meta">
                    Selected candidates proceed to the
                    retraining workflow.
                </div>

            </div>
            """
        )