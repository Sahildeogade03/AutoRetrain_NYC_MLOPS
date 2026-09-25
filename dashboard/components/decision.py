import streamlit as st


def render_decision_breakdown(
    monitoring: dict,
):

    checks = [
        (
            "Drift detected",
            monitoring["drift_detected"],
        ),
        (
            "Recent change",
            monitoring["recent_change"],
        ),
        (
            "Meaningful MAE degradation",
            monitoring[
                "meaningful_degradation"
            ],
        ),
    ]

    for label, passed in checks:

        if passed:
            marker = "✓"
            status = "Satisfied"
        else:
            marker = "○"
            status = "Not satisfied"

        st.html(
            f"""
            <div class="decision-row">
                <div class="decision-marker">
                    {marker}
                </div>
                <div class="decision-label">
                    {label}
                </div>
                <div class="decision-status">
                    {status}
                </div>
            </div>
            """
        )

    if monitoring["retrain_required"]:

        decision = "RETRAINING REQUIRED"

    else:

        decision = "NO RETRAINING REQUIRED"

    st.html(
        f"""
        <div class="decision-result">
            <div class="decision-result-label">
                FINAL DECISION
            </div>
            <div class="decision-result-value">
                {decision}
            </div>
            <div class="decision-result-meta">
                {monitoring["reason"]}
            </div>
        </div>
        """
    )