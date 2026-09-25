import streamlit as st


def render_kpi(
    label: str,
    value: str,
    delta: str = "",
):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-delta">
                {delta}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )