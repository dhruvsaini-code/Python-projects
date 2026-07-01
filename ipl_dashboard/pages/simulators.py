import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.ui import render_header, kpi_card
from services.simulator_service import simulate_innings, predict_first_innings_score, simulate_season
from services.valuation_service import estimate_auction_value
from services.venue_analysis import get_venue_list, get_venue_stats
from services.player_analysis import get_batsman_stats, get_bowler_stats
from services.preprocessing import get_unique_teams
from config.settings import PLOTLY_CONFIG

def _render_match_simulator(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, teams: list, venues: list, plotly_template: str) -> None:
    """Renders the over-by-over Monte Carlo match simulator."""
    st.markdown('<div class="subheader-custom">Monte Carlo Over-by-Over Match Simulator</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    team1 = col1.selectbox("Select Batting Team A:", teams, index=0, key="sim_team_a")
    team2 = col2.selectbox("Select Bowling Team B:", [t for t in teams if t != team1], index=0, key="sim_team_b")
    venue = col3.selectbox("Select Stadium Venue:", venues, index=0, key="sim_venue")
    
    if st.button("🎮 Start Match Simulation", key="start_simulation_btn"):
        v_stats = get_venue_stats(matches_df, deliveries_df, venue)
        
        # 1. Innings 1 Simulation
        inn1 = simulate_innings(team1, team2, v_stats)
        target = inn1[-1]["Runs"] + 1
        
        # 2. Innings 2 Simulation
        inn2 = simulate_innings(team2, team1, v_stats, target=target)
        
        # Plot runs progression
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[x["Over"] for x in inn1], y=[x["Runs"] for x in inn1], name=f"{team1} (Innings 1)", mode="lines+markers", line=dict(color="#e74c3c")))
        fig.add_trace(go.Scatter(x=[x["Over"] for x in inn2], y=[x["Runs"] for x in inn2], name=f"{team2} (Chasing)", mode="lines+markers", line=dict(color="#3498db")))
        fig.add_trace(go.Scatter(x=[0, 20], y=[target, target], name="Target", mode="lines", line=dict(color="green", dash="dash")))
        fig.update_layout(template=plotly_template, title="Simulated Match Runs Progression Timeline", xaxis_title="Overs", yaxis_title="Runs", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Outcome
        chase_runs = inn2[-1]["Runs"]
        if chase_runs >= target:
            st.success(f"🏆 Simulated Winner: **{team2}** (Won by {10 - inn2[-1]['Wickets']} wickets!)")
        else:
            st.success(f"🏆 Simulated Winner: **{team1}** (Defended target, won by {target - 1 - chase_runs} runs!)")

def _render_score_predictor(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, teams: list, venues: list) -> None:
    """Renders the 1st innings score predictor panel."""
    st.markdown('<div class="subheader-custom">1st Innings Projected Score Predictor</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    bat_t = col1.selectbox("Batting Team:", teams, index=0, key="sp_bat")
    bowl_t = col2.selectbox("Bowling Team:", [t for t in teams if t != bat_t], index=0, key="sp_bowl")
    venue = st.selectbox("Stadium Venue:", venues, index=0, key="sp_venue")
    
    col3, col4, col5 = st.columns(3)
    score = col3.number_input("Current Score:", min_value=0, max_value=250, value=75)
    wkts = col4.slider("Wickets Lost:", min_value=0, max_value=9, value=2)
    overs = col5.slider("Overs Completed:", min_value=0.1, max_value=19.5, value=10.0, step=0.1)
    
    if st.button("🎯 Calculate Predicted Score", key="score_predict_btn"):
        v_stats = get_venue_stats(matches_df, deliveries_df, venue)
        avg_score = v_stats.get("avg_1st_innings", 160.0)
        
        pred = predict_first_innings_score(bat_t, bowl_t, venue, score, wkts, overs, avg_score)
        kpi_card("Projected Final Score", f"{pred} runs", "kpi-green")

def _render_season_predictor(teams: list, matches_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders the Season Winner tournament fixture simulator."""
    st.markdown('<div class="subheader-custom">IPL Season Champion Simulator (playoffs fixtures included)</div>', unsafe_allow_html=True)
    st.write("Simulate 200 full tournament season campaigns using historical team win rates.")
    
    if st.button("🏆 Run Season Champion Simulation", key="season_simulate_btn"):
        # Compile head-to-head win distributions matrix mapping
        h2h_map = {}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                h2h_matches = matches_df[((matches_df["team1"] == t1) & (matches_df["team2"] == t2)) | ((matches_df["team1"] == t2) & (matches_df["team2"] == t1))]
                t1_wins = len(h2h_matches[h2h_matches["winner"] == t1])
                t2_wins = len(h2h_matches[h2h_matches["winner"] == t2])
                
                win_prob = (t1_wins / len(h2h_matches)) if len(h2h_matches) > 0 else 0.5
                h2h_map[(t1, t2)] = win_prob
                h2h_map[(t2, t1)] = 1.0 - win_prob
                
        champs = {t: 0 for t in teams}
        for _ in range(200):
            season_champs = simulate_season(teams, h2h_map)
            for k, v in season_champs.items():
                champs[k] += v
                
        df_champs = pd.DataFrame([{"Team": k, "Championships Won": v, "Win Probability (%)": round((v/200)*100, 1)} for k, v in champs.items()]).sort_values("Championships Won", ascending=False)
        
        fig = px.bar(df_champs, x="Win Probability (%)", y="Team", orientation="h", title="Simulated Season Winner Probability Map", color="Win Probability (%)", color_continuous_scale="Viridis", text="Win Probability (%)")
        fig.update_layout(template=plotly_template, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_champs, use_container_width=True)

def _render_auction_valuator(deliveries_df: pd.DataFrame) -> None:
    """Renders the player Auction value estimator."""
    st.markdown('<div class="subheader-custom">Player Estimated Auction Salary Valuator</div>', unsafe_allow_html=True)
    all_players = sorted(list(set(deliveries_df["batsman"].dropna().unique()).union(set(deliveries_df["bowler"].dropna().unique()))))
    player = st.selectbox("Search / Select Player to Value:", all_players)
    
    if st.button("💰 Estimate Auction Market Value", key="auction_value_btn"):
        batting_stats = get_batsman_stats(deliveries_df)
        bowling_stats = get_bowler_stats(deliveries_df)
        
        val, tier = estimate_auction_value(player, batting_stats, bowling_stats)
        
        col1, col2 = st.columns(2)
        with col1:
            kpi_card("Estimated Valuation", f"₹ {val:.2f} Crores", "kpi-purple")
        with col2:
            kpi_card("Salary Performance Tier", tier, "kpi-orange")

def show_simulators_page(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Simulators page layout."""
    render_header("MATCH SIMULATORS & VALUATION ENGINE", "Over-by-over Monte Carlo simulators, score calculators, season champion predictors, and auction estimators")
    
    teams = get_unique_teams(matches_df)
    venues = get_venue_list(matches_df)
    
    tab_match, tab_score, tab_season, tab_auction = st.tabs(["🎮 Match Simulator", "🎯 Score Predictor", "🏆 Season Predictor", "💰 Auction Valuator"])
    with tab_match:
        _render_match_simulator(matches_df, deliveries_df, teams, venues, plotly_template)
    with tab_score:
        _render_score_predictor(matches_df, deliveries_df, teams, venues)
    with tab_season:
        _render_season_predictor(teams, matches_df, plotly_template)
    with tab_auction:
        _render_auction_valuator(deliveries_df)
