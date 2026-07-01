import streamlit as st
import pandas as pd
from typing import List

def get_search_options(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> List[str]:
    """Compiles a list of search matches (teams and players) from the database."""
    from services.preprocessing import get_unique_teams
    teams = get_unique_teams(matches_df)
    
    batsmen = set(deliveries_df["batsman"].dropna().unique())
    bowlers = set(deliveries_df["bowler"].dropna().unique())
    players = sorted(list(batsmen.union(bowlers)))
    
    # Combined options list
    return [""] + teams + players

def render_global_search(search_options: List[str], teams_list: List[str]) -> None:
    """Renders search bar widget and handles session state redirection triggers."""
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    selected = st.selectbox(
        "🔍 Global Search: Type any Player or Team to view analytics instantly...",
        options=search_options,
        index=0,
        key="global_search_input"
    )
    
    if selected:
        if selected in teams_list:
            st.session_state["nav_select"] = "👥 Team Performance"
            st.session_state["h2h_t1"] = selected
            st.toast(f"Opening Team Analytics for {selected}...", icon="👥")
            st.rerun()
        else:
            st.session_state["nav_select"] = "🏏 Player Statistics"
            st.session_state["profile_search_autocomplete"] = selected
            st.toast(f"Opening Player Profile for {selected}...", icon="🏏")
            st.rerun()
