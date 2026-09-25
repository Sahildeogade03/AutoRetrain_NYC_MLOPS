import streamlit as st


LIFECYCLE_STAGES = [
    "Production",
    "Drift Detected",
    "Retraining",
    "Evaluation",
    "Deployment",
]


def render_lifecycle(active_stage: str = "Production"):

    active_index = LIFECYCLE_STAGES.index(active_stage)

    items = []

    for index, stage in enumerate(LIFECYCLE_STAGES):

        if index < active_index:
            class_name = "lifecycle-step complete"
            marker = "●"

        elif index == active_index:
            class_name = "lifecycle-step active"
            marker = "●"

        else:
            class_name = "lifecycle-step"
            marker = "○"

        items.append(
            f"""
            <div class="{class_name}">
                {marker}&nbsp; {stage}
            </div>
            """
        )

        if index < len(LIFECYCLE_STAGES) - 1:

            items.append(
                '<div class="lifecycle-arrow">→</div>'
            )

    lifecycle_html = "".join(items)

    st.markdown(
        f"""
        <div class="lifecycle-card">

            <div class="lifecycle">
                {lifecycle_html}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )