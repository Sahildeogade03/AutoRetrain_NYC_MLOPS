import streamlit as st

from config import APP_NAME, APP_SUBTITLE


def render_header():

    left, right = st.columns([7, 2])

    with left:

        st.markdown(
            f"""
            <div class="dashboard-title">
                {APP_NAME}
            </div>

            <div class="dashboard-subtitle">
                {APP_SUBTITLE}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div style="display:flex;
                        justify-content:flex-end;
                        padding-top:4px;">

                <span class="status-pill">
                    ● SYSTEM HEALTHY
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )