import streamlit as st

from components.lifecycle import render_lifecycle
from components.kpi import render_kpi
from components.decision import render_decision_breakdown

from services.monitoring_service import evaluate_monitoring
from services.retraining_service import get_retraining_status


def render_retraining():

    # --------------------------------------------------------------
    # Load monitoring + retraining state
    # --------------------------------------------------------------

    monitoring = evaluate_monitoring()
    retraining = get_retraining_status()

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Retraining Control Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
            Candidate training, evaluation, quality gating and model promotion
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------------
    # Current state
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Current State</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi(
            "DRIFT",
            "DETECTED"
            if monitoring["drift_detected"]
            else "CLEAR",
            f'{len(monitoring["change_points"])} change point(s)',
        )

    with col2:
        render_kpi(
            "RETRAINING",
            "REQUIRED"
            if monitoring["retrain_required"]
            else "NOT REQUIRED",
            "Monitoring decision",
        )

    with col3:
        render_kpi(
            "MAE DEGRADATION",
            f'{monitoring["mae_degradation"]:.2%}',
            "Recent vs historical",
        )

    with col4:
        render_kpi(
            "LAST RUN",
            (
                str(retraining["last_run"])
                if retraining["last_run"] is not None
                else "NONE"
            ),
            "Retraining execution",
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Retraining lifecycle
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Retraining Lifecycle</div>',
        unsafe_allow_html=True,
    )

    if monitoring["retrain_required"]:
        lifecycle_stage = "Retraining"
    elif monitoring["drift_detected"]:
        lifecycle_stage = "Drift Detected"
    else:
        lifecycle_stage = "Production"

    render_lifecycle(active_stage=lifecycle_stage)

    st.markdown("")

    # --------------------------------------------------------------
    # Decision + workflow
    # --------------------------------------------------------------

    decision_col, workflow_col = st.columns([1, 1.2])

    with decision_col:

        st.markdown(
            '<div class="section-heading">Retraining Decision</div>',
            unsafe_allow_html=True,
        )

        render_decision_breakdown(monitoring)

    with workflow_col:

        st.markdown(
            '<div class="section-heading">Workflow</div>',
            unsafe_allow_html=True,
        )

        workflow_steps = [
            (
                "01",
                "Drift Detection",
                "Monitor production residuals and detect changes in forecast behaviour.",
            ),
            (
                "02",
                "Candidate Training",
                "Train a new forecasting candidate using the retraining pipeline.",
            ),
            (
                "03",
                "Evaluation",
                "Evaluate the candidate against the current production model.",
            ),
            (
                "04",
                "Quality Gate",
                "Require the candidate to satisfy the configured promotion criteria.",
            ),
            (
                "05",
                "Promotion",
                "Promote the candidate only when the quality gate passes.",
            ),
        ]

        for number, title, description in workflow_steps:

            st.html(
                f"""
                <div class="config-card">

                    <div class="config-label">
                        STEP {number}
                    </div>

                    <div class="config-value">
                        {title}
                    </div>

                    <div class="config-meta">
                        {description}
                    </div>

                </div>
                """
            )

            st.markdown("")

    # --------------------------------------------------------------
    # System decision
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">System Decision</div>',
        unsafe_allow_html=True,
    )

    if monitoring["retrain_required"]:

        st.html(
            f"""
            <div class="decision-result">

                <div class="decision-result-label">
                    MONITORING ACTION
                </div>

                <div class="decision-result-value">
                    RETRAINING REQUIRED
                </div>

                <div class="decision-result-meta">
                    {monitoring["reason"]}
                </div>

            </div>
            """
        )

    elif monitoring["drift_detected"]:

        st.html(
            f"""
            <div class="decision-result">

                <div class="decision-result-label">
                    MONITORING ACTION
                </div>

                <div class="decision-result-value">
                    MONITORING CONTINUES
                </div>

                <div class="decision-result-meta">
                    Drift was detected, but the configured retraining
                    conditions are not currently satisfied.
                </div>

            </div>
            """
        )

    else:

        st.html(
            """
            <div class="decision-result">

                <div class="decision-result-label">
                    MONITORING ACTION
                </div>

                <div class="decision-result-value">
                    PRODUCTION MODEL RETAINED
                </div>

                <div class="decision-result-meta">
                    No retraining trigger is currently active.
                </div>

            </div>
            """
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Retraining execution status
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Retraining Execution</div>',
        unsafe_allow_html=True,
    )

    execution_col1, execution_col2, execution_col3 = st.columns(3)

    with execution_col1:

        st.html(
            f"""
            <div class="config-card">

                <div class="config-label">
                    STATUS
                </div>

                <div class="config-value">
                    {retraining["status"]}
                </div>

                <div class="config-meta">
                    Automated retraining lifecycle
                </div>

            </div>
            """
        )

    with execution_col2:

        candidate = (
            retraining["candidate_model"]
            if retraining["candidate_model"] is not None
            else "Not available"
        )

        st.html(
            f"""
            <div class="config-card">

                <div class="config-label">
                    CANDIDATE MODEL
                </div>

                <div class="config-value">
                    {candidate}
                </div>

                <div class="config-meta">
                    Latest candidate from retraining
                </div>

            </div>
            """
        )

    with execution_col3:

        quality_gate = (
            retraining["quality_gate"]
            if retraining["quality_gate"] is not None
            else "Not evaluated"
        )

        st.html(
            f"""
            <div class="config-card">

                <div class="config-label">
                    QUALITY GATE
                </div>

                <div class="config-value">
                    {quality_gate}
                </div>

                <div class="config-meta">
                    Candidate promotion decision
                </div>

            </div>
            """
        )

    st.markdown("")

    # --------------------------------------------------------------
    # Promotion status
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Promotion</div>',
        unsafe_allow_html=True,
    )

    st.html(
        f"""
        <div class="decision-result">

            <div class="decision-result-label">
                PRODUCTION PROMOTION
            </div>

            <div class="decision-result-value">
                NOT AVAILABLE
            </div>

            <div class="decision-result-meta">
                Promotion status will appear here once the retraining
                execution service is connected to the dashboard.
            </div>

        </div>
        """
    )