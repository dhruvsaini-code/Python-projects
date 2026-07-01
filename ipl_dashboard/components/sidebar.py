import streamlit as st
import pandas as pd
from typing import Tuple, List
from services.preprocessing import get_unique_seasons, get_unique_teams, get_unique_venues

def render_theme_selector() -> str:
    """Renders the style selection dropdown in the sidebar."""
    st.sidebar.markdown("### Dashboard Style")
    return st.sidebar.selectbox("Select Theme:", ["Dark Mode", "Light Mode"], key="theme_selector")

def render_sidebar_stats(matches_df: pd.DataFrame) -> None:
    """Renders the general database stats overview card."""
    st.sidebar.markdown("### Dashboard Sidebar Stats")
    total_db_matches = len(matches_df)
    min_season = matches_df["season"].min()
    max_season = matches_df["season"].max()
    st.sidebar.info(
        f"📂 **Total Matches:** {total_db_matches} contests\n"
        f"📅 **Seasons:** {min_season} to {max_season}"
    )

def render_global_filters(matches_df: pd.DataFrame) -> Tuple[str, str, str]:
    """Renders drop-down selectors for filtering the datasets globally."""
    st.sidebar.markdown("### Global Filters")
    
    seasons = ["All"] + [str(s) for s in get_unique_seasons(matches_df)]
    season = st.sidebar.selectbox("Filter Season:", seasons, key="global_season_filter")
    
    teams = ["All"] + get_unique_teams(matches_df)
    team = st.sidebar.selectbox("Filter Team:", teams, key="global_team_filter")
    
    venues = ["All"] + get_unique_venues(matches_df)
    venue = st.sidebar.selectbox("Filter Venue:", venues, key="global_venue_filter")
    
    return season, team, venue

def render_navigation() -> str:
    """Renders the main routing selector links in the sidebar."""
    st.sidebar.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.sidebar.markdown("### Navigation")
    
    pages = [
        "🏠 Home Page", 
        "👥 Team Performance", 
        "🏏 Player Statistics", 
        "🏟️ Venue Insights", 
        "📊 Advanced Analytics",
        "🤖 Match Win Predictor",
        "🧙‍♂️ Fantasy Assistant",
        "🎮 Match Simulators"
    ]
    
    return st.sidebar.radio("Navigate Dashboard Pages:", pages, label_visibility="collapsed")

def apply_global_filters(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, season: str, team: str, venue: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies selected filters to the matches and ball-by-ball deliveries datasets."""
    from services.preprocessing import filter_matches
    filtered_matches = filter_matches(matches_df, season, team, venue)
    filtered_match_ids = filtered_matches["id"].tolist()
    filtered_deliveries = deliveries_df[deliveries_df["match_id"].isin(filtered_match_ids)].copy()
    return filtered_matches, filtered_deliveries
