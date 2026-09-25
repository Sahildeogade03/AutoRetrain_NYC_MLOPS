import streamlit as st


def render_kpi(
    label: str,
    value: str,
    delta: str = "",
):
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )