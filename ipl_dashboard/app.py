import streamlit as st
from config.settings import init_page_config, apply_theme
from services.data_loader import get_combined_data
from components.sidebar import render_theme_selector, render_sidebar_stats, render_global_filters, render_navigation, apply_global_filters
from components.search import get_search_options, render_global_search
from components.ui import render_footer
from services.preprocessing import get_unique_teams

# Initialize page settings
init_page_config()

# Theme and style config
theme_mode = render_theme_selector()
plotly_template = apply_theme(theme_mode)

# Load data files
try:
    matches_df, deliveries_df = get_combined_data()
    data_loaded = True
except Exception as e:
    st.error("Error loading datasets. Verify data files are available in data/ folder.")
    st.exception(e)
    data_loaded = False

if data_loaded:
    # Sidebar stats and global filtering
    render_sidebar_stats(matches_df)
    season, team, venue = render_global_filters(matches_df)
    filtered_matches, filtered_deliveries = apply_global_filters(matches_df, deliveries_df, season, team, venue)
    
    # Autocomplete search option compilation
    search_options = get_search_options(matches_df, deliveries_df)
    teams_list = get_unique_teams(matches_df)
    render_global_search(search_options, teams_list)
    
    # Load Navigation Link Page Router
    page = render_navigation()
    
    if page == "🏠 Home Page":
        from pages.home import show_home_page
        show_home_page(filtered_matches, filtered_deliveries, matches_df, deliveries_df, plotly_template)
    elif page == "👥 Team Performance":
        from pages.team_analysis import show_team_page
        show_team_page(filtered_matches, plotly_template)
    elif page == "🏏 Player Statistics":
        from pages.player_analysis import show_player_page
        show_player_page(filtered_deliveries, plotly_template)
    elif page == "🏟️ Venue Insights":
        from pages.venue_analysis import show_venue_page
        show_venue_page(filtered_matches, filtered_deliveries, plotly_template)
    elif page == "📊 Advanced Analytics":
        from pages.advanced_analytics import show_advanced_page
        show_advanced_page(filtered_matches, filtered_deliveries, plotly_template)
    elif page == "🤖 Match Win Predictor":
        from pages.win_predictor import show_win_predictor_page
        show_win_predictor_page(matches_df, deliveries_df, plotly_template)
    elif page == "🧙‍♂️ Fantasy Assistant":
        from pages.fantasy_assistant import show_fantasy_assistant_page
        show_fantasy_assistant_page(matches_df, deliveries_df)
    elif page == "🎮 Match Simulators":
        from pages.simulators import show_simulators_page
        show_simulators_page(matches_df, deliveries_df, plotly_template)
        
    render_footer()
