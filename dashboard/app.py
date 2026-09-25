import streamlit as st

from config import PAGE_CONFIG

from components.theme import load_css
from components.sidebar import render_sidebar
from components.header import render_header

from views.overview import render_overview
from views.forecast import render_forecast
from views.models import render_models
from views.monitoring import render_monitoring
from views.retraining import render_retraining
from views.experiments import render_experiments


st.set_page_config(**PAGE_CONFIG)

load_css()


def main():

    selected_page = render_sidebar()

    render_header()

    pages = {
        "Overview": render_overview,
        "Forecast": render_forecast,
        "Models": render_models,
        "Monitoring": render_monitoring,
        "Retraining": render_retraining,
        "Experiments": render_experiments,
    }

    pages[selected_page]()


if __name__ == "__main__":
    main()