import pandas as pd
import plotly.graph_objects as go


def actual_vs_forecast_chart(
    data: pd.DataFrame,
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["actual"],
            mode="lines",
            name="Actual",
            line=dict(width=2.2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["forecast"],
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
            title="Demand",
            gridcolor="rgba(128,128,128,0.12)",
            zeroline=False,
        ),

        # Let Streamlit's theme remain the primary visual language.
        template="plotly_white",
    )

    return fig