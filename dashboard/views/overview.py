import streamlit as st

from components.kpi import render_kpi
from components.charts import actual_vs_forecast_chart
from components.lifecycle import render_lifecycle

from services.forecast_service import get_forecast_metrics
from services.model_service import get_production_model


def render_overview():

    metrics = get_forecast_metrics()
    model = get_production_model()

    # ---------------------------------------------------------
    # Production Model
    # ---------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Production Model</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        render_kpi(
            "MAE",
            f"{metrics['mae']:.1f}",
            "Current production window",
        )

    with col2:

        render_kpi(
            "RMSE",
            f"{metrics['rmse']:.1f}",
            "Current production window",
        )

    with col3:

        render_kpi(
            "DRIFT",
            f"{metrics['drift']:.1f}%",
            "Current monitoring window",
        )

    st.markdown("")

    # ---------------------------------------------------------
    # Actual vs Forecast
    # ---------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Actual vs Forecast</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True,
    )

    fig = actual_vs_forecast_chart(
        metrics["forecast_data"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ---------------------------------------------------------
    # Model Lifecycle
    # ---------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Model Lifecycle</div>',
        unsafe_allow_html=True,
    )

    render_lifecycle(
        active_stage="Production"
    )

    st.markdown("")

    # ---------------------------------------------------------
    # Production Model Information
    # ---------------------------------------------------------

    model_col, summary_col = st.columns([1, 2])

    with model_col:

        st.markdown(
            f"""
            <div class="model-card">

                <div class="model-label">
                    PRODUCTION MODEL
                </div>

                <div class="model-name">
                    {model["name"]}
                </div>

                <div class="model-meta">
                    Version {model["version"]}
                    · {model["status"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_col:

        st.markdown(
            """
            <div class="model-card">

                <div class="model-label">
                    SYSTEM SUMMARY
                </div>

                <div class="model-name">
                    Forecasting pipeline operational
                </div>

                <div class="model-meta">
                    Production forecasting and monitoring
                    status will be connected to the
                    AutoRetrain-NYC backend.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )