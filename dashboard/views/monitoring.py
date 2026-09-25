import streamlit as st

from components.kpi import render_kpi
from components.charts import (
    rolling_bias_chart,
    rolling_mae_chart,
    forecast_error_chart
)
from components.decision import render_decision_breakdown

from services.monitoring_service import evaluate_monitoring


def render_monitoring():

    # ------------------------------------------------------------------
    # Load monitoring state
    # ------------------------------------------------------------------

    monitoring = evaluate_monitoring()
    monitoring_df = monitoring["monitoring"]

    # ------------------------------------------------------------------
    # Page header
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Monitoring Decision</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            Forecast performance, change-point detection and retraining signals
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ------------------------------------------------------------------
    # Monitoring KPIs
    # ------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        drift_value = (
            "DETECTED"
            if monitoring["drift_detected"]
            else "CLEAR"
        )

        render_kpi(
            "DRIFT STATUS",
            drift_value,
            f'{len(monitoring["change_points"])} change point(s)',
        )

    with col2:
        retrain_value = (
            "REQUIRED"
            if monitoring["retrain_required"]
            else "NOT REQUIRED"
        )

        render_kpi(
            "RETRAINING",
            retrain_value,
            "Current decision",
        )

    with col3:
        render_kpi(
            "HISTORICAL MAE",
            f'{monitoring["historical_mae"]:.2f}',
            "Monitoring baseline",
        )

    with col4:
        render_kpi(
            "RECENT MAE",
            f'{monitoring["recent_mae"]:.2f}',
            "Last 48 observations",
        )

    st.markdown("")

    # ------------------------------------------------------------------
    # Secondary monitoring metrics
    # ------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        degradation = monitoring["mae_degradation"]

        render_kpi(
            "MAE DEGRADATION",
            f"{degradation:.2%}",
            "Recent vs historical baseline",
        )

    with col2:
        latest_cp = monitoring["latest_change_point"]

        latest_cp_text = (
            latest_cp.strftime("%Y-%m-%d %H:%M")
            if latest_cp is not None
            else "None"
        )

        render_kpi(
            "LATEST CHANGE POINT",
            latest_cp_text,
            "Detected boundary",
        )

    st.markdown("")

    # ------------------------------------------------------------------
    # Rolling bias
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Rolling Bias</div>',
        unsafe_allow_html=True,
    )

    bias_fig = rolling_bias_chart(
        monitoring_df,
        lookback_hours=24 * 14,
        change_points=monitoring["change_points"],
    )

    st.plotly_chart(
        bias_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    st.markdown("")

    # ------------------------------------------------------------------
    # Rolling MAE
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Rolling MAE</div>',
        unsafe_allow_html=True,
    )

    mae_fig = rolling_mae_chart(
        monitoring_df,
        lookback_hours=24 * 14,
        change_points=monitoring["change_points"],
    )

    st.plotly_chart(
        mae_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    st.markdown("")

    # ------------------------------------------------------------------
    # Decision + configuration
    # ------------------------------------------------------------------

    decision_col, config_col = st.columns([1.25, 1])

    with decision_col:

        st.markdown(
            '<div class="section-heading">Retraining Decision</div>',
            unsafe_allow_html=True,
        )

        render_decision_breakdown(monitoring)

    with config_col:

        st.markdown(
            '<div class="section-heading">Monitoring Configuration</div>',
            unsafe_allow_html=True,
        )

        config_items = [
            (
                "ROLLING WINDOW",
                "24 hours",
                "Bias and MAE calculation",
            ),
            (
                "CPD MIN SIZE",
                "48 observations",
                "Minimum change-point segment",
            ),
            (
                "CPD PENALTY",
                "10.0",
                "Change-point detection penalty",
            ),
            (
                "PERSISTENCE",
                "48 observations",
                "Recent change validation",
            ),
            (
                "MAE THRESHOLD",
                "5%",
                "Meaningful degradation",
            ),
        ]

        for label, value, description in config_items:

            st.html(
                f"""
                <div class="config-card">
                    <div class="config-label">
                        {label}
                    </div>

                    <div class="config-value">
                        {value}
                    </div>

                    <div class="config-meta">
                        {description}
                    </div>
                </div>
                """
            )

            st.markdown("")

    # ------------------------------------------------------------------
    # Monitoring coverage
    # ------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Monitoring Coverage</div>',
        unsafe_allow_html=True,
    )

    coverage_col1, coverage_col2, coverage_col3 = st.columns(3)

    start_time = monitoring_df.index.min()
    end_time = monitoring_df.index.max()
    observation_count = len(monitoring_df)

    with coverage_col1:
        render_kpi(
            "START",
            start_time.strftime("%Y-%m-%d %H:%M"),
            "First monitored observation",
        )

    with coverage_col2:
        render_kpi(
            "LATEST",
            end_time.strftime("%Y-%m-%d %H:%M"),
            "Latest monitored observation",
        )

    with coverage_col3:
        render_kpi(
            "OBSERVATIONS",
            f"{observation_count:,}",
            "System-level hourly observations",
        )