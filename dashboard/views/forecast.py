import streamlit as st

from components.kpi import render_kpi
from components.charts import (
    actual_vs_forecast_chart,
    forecast_error_chart,
)

from services.forecast_service import get_forecast_metrics
from services.model_service import get_production_model


def render_forecast():

    # --------------------------------------------------------------
    # Load forecast data
    # --------------------------------------------------------------

    forecast = get_forecast_metrics()
    model = get_production_model()

    data = forecast["forecast_data"]

    # --------------------------------------------------------------
    # Page header
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Forecast Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            Production demand forecasts and forecast error analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------------
    # Forecast KPIs
    # --------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi(
            "MAE",
            f'{forecast["mae"]:.2f}',
            "System-level forecast error",
        )

    with col2:
        render_kpi(
            "RMSE",
            f'{forecast["rmse"]:.2f}',
            "System-level forecast error",
        )

    with col3:
        render_kpi(
            "OBSERVATIONS",
            f'{forecast["observation_count"]:,}',
            "Hourly observations",
        )

    with col4:
        render_kpi(
            "LATEST",
            forecast["latest_timestamp"].strftime("%Y-%m-%d %H:%M"),
            "Latest forecast",
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Actual vs Forecast
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Actual vs Forecast</div>',
        unsafe_allow_html=True,
    )

    fig = actual_vs_forecast_chart(
        data,
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

    st.markdown("")

    # --------------------------------------------------------------
    # Forecast Error
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Forecast Error</div>',
        unsafe_allow_html=True,
    )

    error_fig = forecast_error_chart(
        data,
        lookback_hours=24 * 7,
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    st.markdown("")

    # --------------------------------------------------------------
    # Production model information
    # --------------------------------------------------------------

    model_col, metadata_col = st.columns([1, 2])

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

    with metadata_col:

        st.html(
            f"""
            <div class="model-card">
                <div class="model-label">
                    FORECAST SUMMARY
                </div>

                <div class="model-name">
                    {forecast["observation_count"]:,} hourly observations
                </div>

                <div class="model-meta">
                    Monitoring window:
                    {data["timestamp"].min().strftime("%Y-%m-%d %H:%M")}
                    →
                    {data["timestamp"].max().strftime("%Y-%m-%d %H:%M")}
                </div>

                <div class="model-meta">
                    Latest forecast:
                    {forecast["latest_timestamp"].strftime("%Y-%m-%d %H:%M")}
                </div>
            </div>
            """
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Forecast data
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Forecast Data</div>',
        unsafe_allow_html=True,
    )

    display_data = data[
        [
            "timestamp",
            "actual_demand",
            "forecast_demand",
            "residual",
        ]
    ].copy()

    display_data = display_data.sort_values(
        "timestamp",
        ascending=False,
    )

    display_data["timestamp"] = display_data["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    display_data = display_data.rename(
        columns={
            "timestamp": "Timestamp",
            "actual_demand": "Actual Demand",
            "forecast_demand": "Forecast",
            "residual": "Residual",
        }
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )