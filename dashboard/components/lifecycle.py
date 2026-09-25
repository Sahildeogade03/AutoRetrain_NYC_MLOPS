import streamlit as st


LIFECYCLE_STAGES = [
    "Production",
    "Drift Detected",
    "Retraining",
    "Evaluation",
    "Deployment",
]


def render_lifecycle(active_stage: str = "Production"):

    if active_stage not in LIFECYCLE_STAGES:
        active_stage = "Production"

    active_index = LIFECYCLE_STAGES.index(
        active_stage
    )

    items = []

    for index, stage in enumerate(
        LIFECYCLE_STAGES
    ):

        if index < active_index:
            class_name = "lifecycle-step complete"
            marker = "✓"

        elif index == active_index:
            class_name = "lifecycle-step active"
            marker = "●"

        else:
            class_name = "lifecycle-step"
            marker = "○"

        items.append(
            f'<div class="{class_name}">'
            f'<span>{marker}</span>'
            f'<span>{stage}</span>'
            f'</div>'
        )

        if index < len(LIFECYCLE_STAGES) - 1:
            items.append(
                '<div class="lifecycle-arrow">→</div>'
            )

    html = (
        '<div class="lifecycle-card">'
        '<div class="lifecycle">'
        + "".join(items)
        + "</div>"
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )