import streamlit as st


def render_monitoring_summary(monitoring: dict):

    drift_detected = monitoring["drift_detected"]
    retrain_required = monitoring["retrain_required"]

    historical_mae = monitoring["historical_mae"]
    recent_mae = monitoring["recent_mae"]
    degradation = monitoring["mae_degradation"]

    latest_cp = monitoring["latest_change_point"]

    drift_status = "Detected" if drift_detected else "Clear"

    retraining_status = (
        "Required"
        if retrain_required
        else "Not required"
    )

    latest_cp_text = (
        latest_cp.strftime("%Y-%m-%d %H:%M")
        if latest_cp is not None
        else "None"
    )

    cards = [
        (
            "DRIFT STATUS",
            drift_status,
            "Change-point monitoring",
        ),
        (
            "RETRAINING",
            retraining_status,
            "Current decision",
        ),
        (
            "HISTORICAL MAE",
            f"{historical_mae:.2f}",
            "Monitoring baseline",
        ),
        (
            "RECENT MAE",
            f"{recent_mae:.2f}",
            "Last 48 observations",
        ),
        (
            "MAE DEGRADATION",
            f"{degradation:.2%}",
            "Against historical baseline",
        ),
        (
            "LATEST CHANGE POINT",
            latest_cp_text,
            "Detected boundary",
        ),
    ]

    for row_start in range(0, len(cards), 3):

        columns = st.columns(3)

        for column, card in zip(
            columns,
            cards[row_start:row_start + 3],
        ):

            label, value, description = card

            with column:
                st.html(
                    f"""
                    <div class="monitoring-card">
                        <div class="monitoring-label">
                            {label}
                        </div>

                        <div class="monitoring-value">
                            {value}
                        </div>

                        <div class="monitoring-meta">
                            {description}
                        </div>
                    </div>
                    """
                )

        if row_start == 0:
            st.markdown("")