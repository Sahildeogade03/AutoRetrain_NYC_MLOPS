import streamlit as st

from config import APP_NAME, PAGES


PAGE_ICONS = {
    "Overview": "⌂",
    "Forecast": "⌁",
    "Models": "◈",
    "Monitoring": "◉",
    "Retraining": "↻",
    "Experiments": "⌘",
}


def render_sidebar() -> str:

    with st.sidebar:

        # --------------------------------------------------
        # Brand
        # --------------------------------------------------

        st.markdown(
            f"""
            <div class="sidebar-brand">
                🚕 {APP_NAME}
            </div>

            <div class="sidebar-subtitle">
                ML Operations Control Center
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # Navigation
        # --------------------------------------------------

        st.markdown(
            '<div class="sidebar-section-label">WORKSPACE</div>',
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Navigation",
            PAGES,
            format_func=lambda page: (
                f"{PAGE_ICONS.get(page, '•')}    {page}"
            ),
            label_visibility="collapsed",
        )

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # System status
        # --------------------------------------------------

        st.markdown(
            '<div class="sidebar-section-label">SYSTEM</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-status-card">

                <div class="sidebar-status-row">
                    <span class="status-indicator">●</span>
                    <span>System Healthy</span>
                </div>

                <div class="sidebar-status-meta">
                    Production environment
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # Environment
        # --------------------------------------------------

        st.markdown(
            """
            <div class="sidebar-environment">
                <span>Environment</span>
                <strong>production</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-version">
                AutoRetrain-NYC · v0.1.0
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page