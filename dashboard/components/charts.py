import pandas as pd
import plotly.graph_objects as go


def actual_vs_forecast_chart(
    data: pd.DataFrame,
    lookback_hours: int = 24 * 7,
) -> go.Figure:

    data = data.sort_values("timestamp")

    if len(data) > lookback_hours:
        data = data.tail(lookback_hours)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["actual_demand"],
            mode="lines",
            name="Actual",
            line=dict(width=2.2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["forecast_demand"],
            mode="lines",
            name="Forecast",
            line=dict(
                width=2,
                dash="dash",
            ),
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            showgrid=False,
            title=None,
        ),
        yaxis=dict(
            title="Hourly Demand",
            gridcolor=(
                "rgba(128,128,128,0.12)"
            ),
            zeroline=False,
        ),
        template="plotly_white",
    )

    return fig


def rolling_bias_chart(
    monitoring: pd.DataFrame,
    lookback_hours: int = 24 * 14,
    change_points: list[int] | None = None,
) -> go.Figure:

    data = monitoring.copy()

    if len(data) > lookback_hours:
        data = data.tail(lookback_hours)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["rolling_bias"],
            mode="lines",
            name="Rolling Bias",
            line=dict(width=2),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_width=1,
    )

    if change_points:

        for cp in change_points:

            if cp <= len(monitoring):

                position = min(
                    max(cp - 1, 0),
                    len(monitoring) - 1,
                )

                timestamp = monitoring.index[
                    position
                ]

                if timestamp >= data.index.min():

                    fig.add_vline(
                        x=timestamp,
                        line_dash="dash",
                        line_width=1,
                    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            title=None,
        ),
        yaxis=dict(
            title="Bias",
            gridcolor=(
                "rgba(128,128,128,0.12)"
            ),
            zeroline=False,
        ),
        template="plotly_white",
    )

    return fig


def rolling_mae_chart(
    monitoring: pd.DataFrame,
    lookback_hours: int = 24 * 14,
    change_points: list[int] | None = None,
) -> go.Figure:

    data = monitoring.copy()

    if len(data) > lookback_hours:
        data = data.tail(lookback_hours)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["rolling_mae"],
            mode="lines",
            name="Rolling MAE",
            line=dict(width=2),
        )
    )

    if change_points:

        for cp in change_points:

            if cp <= len(monitoring):

                position = min(
                    max(cp - 1, 0),
                    len(monitoring) - 1,
                )

                timestamp = monitoring.index[
                    position
                ]

                if timestamp >= data.index.min():

                    fig.add_vline(
                        x=timestamp,
                        line_dash="dash",
                        line_width=1,
                    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            title=None,
        ),
        yaxis=dict(
            title="MAE",
            gridcolor=(
                "rgba(128,128,128,0.12)"
            ),
            zeroline=False,
        ),
        template="plotly_white",
    )

    return fig


def forecast_error_chart(data, lookback_hours=24 * 7):
    plot_data = data.copy()

    if len(plot_data) > lookback_hours:
        plot_data = plot_data.tail(lookback_hours)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_data["timestamp"],
            y=plot_data["residual"],
            mode="lines",
            name="Forecast Error",
            line=dict(width=2),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_width=1,
    )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=15, b=10),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            title=None,
        ),
        yaxis=dict(
            title="Residual",
            gridcolor="rgba(128,128,128,0.12)",
            zeroline=False,
        ),
        template="plotly_white",
    )

    return fig