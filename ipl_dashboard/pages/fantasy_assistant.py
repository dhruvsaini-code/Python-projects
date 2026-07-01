import streamlit as st
import pandas as pd
from components.ui import render_header, kpi_card
from services.fantasy_engine import get_fantasy_points, classify_player_roles, get_optimized_squad
from services.valuation_service import find_similar_players
from services.preprocessing import get_unique_teams

def _render_dream11_section(fantasy_df: pd.DataFrame, roles: dict, teams_list: list) -> None:
    """Renders the Dream11 optimized lineup suggestions section."""
    st.markdown('<div class="subheader-custom">Dream11 Optimized Squad Generator</div>', unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    team_a = col_t1.selectbox("Select Team A:", teams_list, index=0)
    team_b = col_t2.selectbox("Select Team B:", teams_list, index=1 if len(teams_list) > 1 else 0)
    
    if team_a == team_b:
        st.warning("Please select two different teams.")
        return
        
    budget = st.slider("Select Credit Budget Limit:", min_value=80.0, max_value=100.0, value=100.0, step=0.5)
    
    if st.button("🧙‍♂️ Generate Dream11 Optimized Lineup", key="dream11_generate_btn"):
        squad_df, captains = get_optimized_squad(fantasy_df, team_a, team_b, roles, budget)
        
        if squad_df.empty:
            st.warning("Unable to generate squad matching constraints under these filters.")
            return
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            kpi_card("Captain Pick (2x Points)", captains["Captain"], "kpi-green")
        with col_c2:
            kpi_card("Vice-Captain Pick (1.5x Points)", captains["Vice-Captain"], "kpi-orange")
            
        st.write("Suggested 11-Player Lineup")
        st.dataframe(squad_df.style.background_gradient(subset=["Avg Points"], cmap="Blues"), use_container_width=True)

def _render_similarity_section(fantasy_df: pd.DataFrame) -> None:
    """Renders the Player Similarity Finder page tab."""
    st.markdown('<div class="subheader-custom">Player Similarity Recommendation Engine</div>', unsafe_allow_html=True)
    
    all_players = sorted(fantasy_df["Player"].tolist())
    target_player = st.selectbox("Search Target Player to find matches:", all_players)
    player_type = st.radio("Player Specialization Role:", ["batsman", "bowler"], horizontal=True)
    
    if st.button("🔍 Find Similar Players", key="similarity_finder_btn"):
        matches = find_similar_players(target_player, fantasy_df, player_type, limit=5)
        
        if not matches:
            st.info(f"No similar players found for {target_player} in this dataset.")
            return
            
        st.success(f"Top 5 Player Matches for {target_player} ({player_type.title()} profile):")
        df_matches = pd.DataFrame(matches, columns=["Player Match Name", "Cosine Similarity Score (%)"])
        st.dataframe(df_matches.style.background_gradient(subset=["Cosine Similarity Score (%)"], cmap="Greens"), use_container_width=True)

def show_fantasy_assistant_page(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> None:
    """Renders the entire Fantasy Assistant page layout."""
    render_header("FANTASY CRICKET ASSISTANT", "Dream11 optimized lineup suggestions, captains ranking, and player similarity matches")
    
    fantasy_df = get_fantasy_points(deliveries_df)
    roles = classify_player_roles(deliveries_df)
    teams_list = get_unique_teams(matches_df)
    
    tab_dream11, tab_similarity = st.tabs(["🧙‍♂️ Dream11 Squad Suggestions", "👥 Player Similarity Finder"])
    with tab_dream11:
        _render_dream11_section(fantasy_df, roles, teams_list)
    with tab_similarity:
        _render_similarity_section(fantasy_df)
