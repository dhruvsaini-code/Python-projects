import os
import streamlit as st
from constants.paths import CSS_PATH

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'ipl_analytics_chart',
        'height': 550,
        'width': 800,
        'scale': 2
    }
}

def init_page_config() -> None:
    """Configures the main Streamlit page settings."""
    st.set_page_config(
        page_title="IPL Data Analytics & Match Predictor",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_theme(theme_mode: str) -> str:
    """
    Injects CSS variables based on theme selection.
    Returns the string representing the matching Plotly template name.
    """
    if theme_mode == "Light Mode":
        st.markdown(
            """
            <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
                --card-bg: rgba(0, 0, 0, 0.02);
                --card-hover-bg: rgba(0, 0, 0, 0.05);
                --border-color: rgba(0, 0, 0, 0.08);
                --border-hover-color: rgba(0, 0, 0, 0.15);
                --text-color: #111827;
                --text-muted: rgba(0, 0, 0, 0.6);
                --shadow-color: rgba(0, 0, 0, 0.05);
                --shadow-hover-color: rgba(0, 0, 0, 0.08);
                --primary-gradient: linear-gradient(90deg, #7c3aed 0%, #0891b2 50%, #2563eb 100%);
                --primary-color: #2563eb;
                --accent-color: #7c3aed;
                --sidebar-bg: #f3f4f6;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        plotly_template = "plotly_white"
    else:
        st.markdown(
            """
            <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #0e1117 0%, #1e293b 100%);
                --card-bg: rgba(255, 255, 255, 0.05);
                --card-hover-bg: rgba(255, 255, 255, 0.08);
                --border-color: rgba(255, 255, 255, 0.1);
                --border-hover-color: rgba(255, 255, 255, 0.2);
                --text-color: #f3f4f6;
                --text-muted: rgba(255, 255, 255, 0.7);
                --shadow-color: rgba(0, 0, 0, 0.3);
                --shadow-hover-color: rgba(255, 255, 255, 0.03);
                --primary-gradient: linear-gradient(90deg, #a855f7 0%, #00f2fe 50%, #4facfe 100%);
                --primary-color: #00f2fe;
                --accent-color: #a855f7;
                --sidebar-bg: #0e1117;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        plotly_template = "plotly_dark"

    # Load custom stylesheet
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom stylesheet assets/custom.css not found. Standard styling applied.")

    return plotly_template
