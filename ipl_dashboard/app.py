import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Configure streamlit page settings (MUST be first streamlit call)
st.set_page_config(
    page_title="IPL Data Analytics & Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
from src.data_loader import get_combined_data, check_and_download_data
from src.preprocessing import filter_matches
import src.team_analysis as ta
import src.player_analysis as pa
import src.venue_analysis as va
from src.win_predictor import get_or_train_predictor, predict_match_outcome

# Load Custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom stylesheet assets/custom.css not found. Standard styling applied.")

# Header banner helper
def render_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p style="text-align: center; color: rgba(255,255,255,0.7); margin-top:-25px; font-size:1.15rem; margin-bottom: 25px;">{subtitle}</p>', unsafe_allow_html=True)

# Helper function to create premium glass-styled KPI cards
def kpi_card(title: str, value: str, style_class: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value {style_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Load data
try:
    matches_df, deliveries_df = get_combined_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading datasets. Please make sure matches.csv and deliveries.csv are available in the data/ folder.")
    st.exception(e)
    data_loaded = False

if data_loaded:
    # Sidebar navigation
    st.sidebar.markdown("# IPL Analytics Hub")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home Page", 
            "👥 Team Performance", 
            "🏏 Player Statistics", 
            "🏟️ Venue Insights", 
            "📊 Advanced Analytics",
            "🤖 Match Win Predictor"
        ]
    )
    
    st.sidebar.markdown('<hr style="border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;">', unsafe_allow_html=True)
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides interactive analytics of IPL matches (2008-2019/2020) "
        "and a Machine Learning Win Predictor for live second-innings chases."
    )
    
    # ----------------------------------------------------
    # HOME PAGE
    # ----------------------------------------------------
    if page == "🏠 Home Page":
        render_header("IPL DATA ANALYTICS & WIN PREDICTOR", "Interactive Analysis Dashboard & Machine Learning Match outcome predictor")
        
        # Calculate stats
        total_matches = len(matches_df)
        total_seasons = matches_df["season"].nunique()
        total_venues = matches_df["venue"].nunique()
        
        # Extract unique players
        batsmen = set(deliveries_df["batsman"].unique())
        bowlers = set(deliveries_df["bowler"].unique())
        total_players = len(batsmen.union(bowlers))
        
        # Display KPI cards in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Total Matches", f"{total_matches}", "kpi-green")
        with col2:
            kpi_card("Seasons Covered", f"{total_seasons}", "kpi-orange")
        with col3:
            kpi_card("Unique Players", f"{total_players}", "kpi-purple")
        with col4:
            kpi_card("Venues Host", f"{total_venues}")
            
        st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
        
        # Recent matches table / season selector
        col_s, col_t = st.columns([1, 2])
        with col_s:
            st.markdown('<div class="subheader-custom">IPL General Overview</div>', unsafe_allow_html=True)
            st.write(
                "Welcome to the ultimate IPL Analytics dashboard. Browse the sidebar pages to explore detailed team "
                "histories, player leaderboards, venue characteristics, phase-by-phase runs distribution, and "
                "evaluate live match chase probabilities using machine learning models."
            )
            
            # Simple chart of matches per season
            matches_per_season = matches_df["season"].value_counts().reset_index().rename(columns={"count": "Matches"})
            matches_per_season.columns = ["Season", "Matches"]
            fig_seasons = px.bar(
                matches_per_season, x="Season", y="Matches", 
                title="Matches Played per Season",
                color="Matches", color_continuous_scale="Agsunset"
            )
            fig_seasons.update_layout(template="plotly_dark", height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_seasons, use_container_width=True)
            
        with col_t:
            st.markdown('<div class="subheader-custom">Recent IPL Matches & Winners</div>', unsafe_allow_html=True)
            
            season_list = sorted(matches_df["season"].unique().tolist(), reverse=True)
            selected_season = st.selectbox("Filter Matches by Season:", season_list)
            
            season_df = matches_df[matches_df["season"] == selected_season][["date", "team1", "team2", "venue", "toss_winner", "toss_decision", "winner", "result"]]
            
            # Beautify table
            st.dataframe(season_df.style.highlight_max(subset=["winner"], color="rgba(46, 204, 113, 0.2)"), height=300, use_container_width=True)
            
            # Download report as CSV button
            csv_data = season_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Season Report as CSV",
                data=csv_data,
                file_name=f"IPL_{selected_season}_Season_Report.csv",
                mime="text/csv",
            )
            
    # ----------------------------------------------------
    # TEAM PERFORMANCE PAGE
    # ----------------------------------------------------
    elif page == "👥 Team Performance":
        render_header("TEAM ANALYSIS", "Inspect win histories, season progressions, and head-to-head records")
        
        tab1, tab2 = st.tabs(["📊 Team Win Analytics", "⚔️ Head-to-Head Comparison"])
        
        with tab1:
            st.markdown('<div class="subheader-custom">Overall Team Performances</div>', unsafe_allow_html=True)
            
            col_l, col_r = st.columns([3, 2])
            with col_l:
                # Plotly overall wins
                fig_wins = ta.plot_team_wins(matches_df)
                st.plotly_chart(fig_wins, use_container_width=True)
            with col_r:
                # Team Win statistics table
                team_stats = ta.get_team_stats(matches_df)
                st.write("Team Win Percentage Leaderboard")
                st.dataframe(team_stats.style.background_gradient(subset=["Win Percentage (%)"], cmap="Greens"), use_container_width=True, height=400)
                
                # Download statistics as CSV
                csv_stats = team_stats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Team Stats as CSV",
                    data=csv_stats,
                    file_name="IPL_Team_Win_Statistics.csv",
                    mime="text/csv"
                )
                
            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            
            st.markdown('<div class="subheader-custom">Season-wise Progression & Toss Analytics</div>', unsafe_allow_html=True)
            
            teams_list = ta.get_team_stats(matches_df)["Team"].tolist()
            selected_team = st.selectbox("Select Team to Analyze Progression:", teams_list)
            
            col_prog_l, col_prog_r = st.columns(2)
            with col_prog_l:
                fig_prog = ta.plot_season_performance(matches_df, selected_team)
                st.plotly_chart(fig_prog, use_container_width=True)
            with col_prog_r:
                col_toss_l, col_toss_r = st.columns(2)
                with col_toss_l:
                    fig_toss_dec = ta.plot_toss_decisions(matches_df, selected_team)
                    st.plotly_chart(fig_toss_dec, use_container_width=True)
                with col_toss_r:
                    fig_toss_imp = ta.plot_toss_impact(matches_df, selected_team)
                    st.plotly_chart(fig_toss_imp, use_container_width=True)
                    
        with tab2:
            st.markdown('<div class="subheader-custom">Compare Head-to-Head Statistics</div>', unsafe_allow_html=True)
            
            teams_list = ta.get_team_stats(matches_df)["Team"].tolist()
            
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                team1 = st.selectbox("Select First Team:", teams_list, index=0)
            with col_sel2:
                # Set second team default to index 1
                team2 = st.selectbox("Select Second Team:", teams_list, index=1 if len(teams_list) > 1 else 0)
                
            if team1 == team2:
                st.warning("Please select two different teams for head-to-head comparison.")
            else:
                h2h = ta.get_head_to_head_stats(matches_df, team1, team2)
                
                if h2h["total"] == 0:
                    st.info(f"No match played between {team1} and {team2} yet.")
                else:
                    # Beautiful metrics row
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        kpi_card(f"{team1} Wins", f"{h2h['team1_wins']}", "kpi-green")
                    with col_m2:
                        kpi_card("Total Contests", f"{h2h['total']}", "kpi-orange")
                    with col_m3:
                        kpi_card(f"{team2} Wins", f"{h2h['team2_wins']}", "kpi-purple")
                        
                    # Visual head-to-head comparison donut
                    labels = [team1, team2]
                    values = [h2h["team1_wins"], h2h["team2_wins"]]
                    if h2h["no_result"] > 0:
                        labels.append("No Result/Tie")
                        values.append(h2h["no_result"])
                        
                    fig_h2h = px.pie(
                        names=labels,
                        values=values,
                        title=f"{team1} vs {team2} - Wins Distribution",
                        color_discrete_sequence=["#2980b9", "#8e44ad", "#7f8c8d"],
                        hole=0.4
                    )
                    fig_h2h.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_h2h, use_container_width=True)
                    
                    # List matches between the two
                    st.markdown("##### Match Logs")
                    h2h_matches_df = matches_df[
                        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
                        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
                    ][["season", "date", "venue", "toss_winner", "toss_decision", "winner", "result"]]
                    st.dataframe(h2h_matches_df, use_container_width=True)
                    
    # ----------------------------------------------------
    # PLAYER STATISTICS PAGE
    # ----------------------------------------------------
    elif page == "🏏 Player Statistics":
        render_header("PLAYER LEADERBOARD & PROFILE", "Detailed batsman & bowler stats, caps progression, and player search")
        
        tab1, tab2, tab3 = st.tabs(["🏆 Player Leaderboards", "🍊 Cap Trends (Orange & Purple)", "🔍 Player Search Profile"])
        
        # Cache standard stats to make loading snappier
        batsmen_df = pa.get_batsman_stats(deliveries_df)
        bowlers_df = pa.get_bowler_stats(deliveries_df)
        
        with tab1:
            st.markdown('<div class="subheader-custom">IPL Batting Leaderboard</div>', unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([3, 2])
            with col_b1:
                # Plotly top 10 batsmen
                fig_top_bat = pa.plot_top_batsmen(batsmen_df, limit=10)
                st.plotly_chart(fig_top_bat, use_container_width=True)
            with col_b2:
                # Batsmen stats list (filterable)
                min_runs = st.slider("Minimum Runs Filter:", min_value=0, max_value=5000, value=1000, step=100)
                filtered_batsmen = batsmen_df[batsmen_df["Runs"] >= min_runs].sort_values("Strike Rate", ascending=False)
                st.write("Highest Strike Rates (filtered by Min Runs)")
                st.dataframe(filtered_batsmen[["Player", "Innings", "Runs", "Strike Rate", "Average"]].style.background_gradient(subset=["Runs", "Strike Rate"], cmap="Oranges"), use_container_width=True, height=300)
                
            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            
            st.markdown('<div class="subheader-custom">IPL Bowling Leaderboard</div>', unsafe_allow_html=True)
            col_bo1, col_bo2 = st.columns([3, 2])
            with col_bo1:
                # Plotly top 10 bowlers
                fig_top_bow = pa.plot_top_bowlers(bowlers_df, limit=10)
                st.plotly_chart(fig_top_bow, use_container_width=True)
            with col_bo2:
                # Bowlers stats list
                min_wickets = st.slider("Minimum Wickets Filter:", min_value=0, max_value=150, value=30, step=5)
                filtered_bowlers = bowlers_df[bowlers_df["Wickets"] >= min_wickets].sort_values("Economy", ascending=True)
                st.write("Best Economy Rates (filtered by Min Wickets)")
                st.dataframe(filtered_bowlers[["Player", "Innings", "Wickets", "Overs", "Economy", "Average"]].style.background_gradient(subset=["Wickets"], cmap="Purples"), use_container_width=True, height=300)
                
        with tab2:
            st.markdown('<div class="subheader-custom">Orange & Purple Cap Winners over Seasons</div>', unsafe_allow_html=True)
            
            orange_df = pa.get_orange_cap_trends(matches_df, deliveries_df)
            purple_df = pa.get_purple_cap_trends(matches_df, deliveries_df)
            
            col_cap_l, col_cap_r = st.columns(2)
            with col_cap_l:
                st.markdown("##### 🍊 Orange Cap progression")
                # Plotly line chart runs progression
                fig_orange = px.line(
                    orange_df, x="season", y="Runs", text="Player", 
                    title="Orange Cap Winner Runs per Season",
                    markers=True, line_shape="linear"
                )
                fig_orange.update_layout(template="plotly_dark", height=400)
                fig_orange.update_traces(textposition="top center", line_color="#ff9900", marker=dict(size=8, color="#d35400"))
                st.plotly_chart(fig_orange, use_container_width=True)
                st.dataframe(orange_df, use_container_width=True)
                
            with col_cap_r:
                st.markdown("##### 🍇 Purple Cap progression")
                # Plotly line chart wickets progression
                fig_purple = px.line(
                    purple_df, x="season", y="Wickets", text="Player", 
                    title="Purple Cap Winner Wickets per Season",
                    markers=True
                )
                fig_purple.update_layout(template="plotly_dark", height=400)
                fig_purple.update_traces(textposition="top center", line_color="#8e44ad", marker=dict(size=8, color="#9b59b6"))
                st.plotly_chart(fig_purple, use_container_width=True)
                st.dataframe(purple_df, use_container_width=True)
                
        with tab3:
            st.markdown('<div class="subheader-custom">Search Player Analytics Profile</div>', unsafe_allow_html=True)
            
            # Combine all players for autocomplete search
            all_players = sorted(list(set(deliveries_df["batsman"].dropna().unique()).union(set(deliveries_df["bowler"].dropna().unique()))))
            search_query = st.selectbox("Search / Select Player:", all_players)
            
            if search_query:
                profile = pa.get_player_profile(search_query, deliveries_df)
                
                st.markdown(f"### 👤 {profile['name']} - Profile Overview")
                
                # Render batting profile if player has batted
                if profile["has_batted"]:
                    st.markdown('<div class="subheader-custom" style="font-size:1.2rem; margin-top:10px;">🏏 Batting Metrics</div>', unsafe_allow_html=True)
                    bp = profile["batting"]
                    
                    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
                    with col_b1:
                        kpi_card("Runs", f"{bp['Runs']}", "kpi-orange")
                    with col_b2:
                        kpi_card("Strike Rate", f"{bp['Strike Rate']}")
                    with col_b3:
                        kpi_card("Average", f"{bp['Average']}")
                    with col_b4:
                        kpi_card("Highest Score", f"{bp['Highest Score']}")
                    with col_b5:
                        kpi_card("50s / 100s", f"{bp['50s']} / {bp['100s']}")
                        
                    st.write(f"Innings Played: **{bp['Innings']}** | Balls Faced: **{bp['Balls Faced']}** | Fours (4s): **{bp['Fours']}** | Sixes (6s): **{bp['Sixes']}**")
                    
                # Render bowling profile if player has bowled
                if profile["has_bowled"]:
                    st.markdown('<div class="subheader-custom" style="font-size:1.2rem; margin-top:20px;">⚾ Bowling Metrics</div>', unsafe_allow_html=True)
                    bop = profile["bowling"]
                    
                    col_bo1, col_bo2, col_bo3, col_bo4, col_bo5 = st.columns(5)
                    with col_bo1:
                        kpi_card("Wickets", f"{bop['Wickets']}", "kpi-purple")
                    with col_bo2:
                        kpi_card("Economy", f"{bop['Economy']}")
                    with col_bo3:
                        kpi_card("Average", f"{bop['Average']}")
                    with col_bo4:
                        kpi_card("Best Figures", f"{bop['Best Figures']}")
                    with col_bo5:
                        kpi_card("3W / 5W", f"{bop['3W Hauls']} / {bop['5W Hauls']}")
                        
                    st.write(f"Innings Bowled: **{bop['Innings']}** | Overs Bowled: **{bop['Overs']}** | Strike Rate: **{bop['Strike Rate']}**")
                    
                if not profile["has_batted"] and not profile["has_bowled"]:
                    st.warning("No records found for this player.")
                    
    # ----------------------------------------------------
    # VENUE INSIGHTS PAGE
    # ----------------------------------------------------
    elif page == "🏟️ Venue Insights":
        render_header("VENUE STATS & ANALYTICS", "Understand the characteristics, average scores, and toss biases of IPL stadiums")
        
        tab1, tab2 = st.tabs(["🏟️ Venue Specifics", "📊 Comparative Venue Metrics"])
        
        with tab1:
            st.markdown('<div class="subheader-custom">Analyze Individual Venue Details</div>', unsafe_allow_html=True)
            
            venues = va.get_venue_list(matches_df)
            selected_venue = st.selectbox("Select Venue to Analyze:", venues)
            
            if selected_venue:
                v_stats = va.get_venue_stats(matches_df, deliveries_df, selected_venue)
                
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    kpi_card("Matches Hosted", f"{v_stats['total_matches']}", "kpi-green")
                with col_v2:
                    kpi_card("Avg 1st Innings Score", f"{v_stats['avg_1st_innings']}", "kpi-orange")
                with col_v3:
                    kpi_card("Toss Winner Match Win (%)", f"{v_stats['toss_win_match_win_pct']}%", "kpi-purple")
                    
                col_vl, col_vr = st.columns(2)
                with col_vl:
                    # Pie chart outcomes
                    fig_chase = va.plot_venue_chasing_vs_defending(v_stats)
                    st.plotly_chart(fig_chase, use_container_width=True)
                with col_vr:
                    # Show detailed high/low stats
                    st.markdown('<div class="subheader-custom" style="font-size:1.15rem; margin-top:20px;">Highest & Lowest Innings Totals</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="margin-bottom: 15px;">
                            <div style="color: #ffaa00; font-weight:600; text-transform:uppercase;">Highest Innings Total</div>
                            <div style="font-size: 1.8rem; font-weight:800; color:#fff;">{v_stats['highest_score']} runs</div>
                            <div style="color:rgba(255,255,255,0.7); font-size:0.95rem;">Scored by: <b>{v_stats['highest_team']}</b></div>
                        </div>
                        <div class="kpi-card">
                            <div style="color: #a855f7; font-weight:600; text-transform:uppercase;">Lowest Innings Total</div>
                            <div style="font-size: 1.8rem; font-weight:800; color:#fff;">{v_stats['lowest_score']} runs</div>
                            <div style="color:rgba(255,255,255,0.7); font-size:0.95rem;">Scored by: <b>{v_stats['lowest_team']}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Quick notes
                    chase_success = v_stats['chasing_wins'] / v_stats['total_matches'] if v_stats['total_matches'] > 0 else 0
                    if chase_success > 0.55:
                        st.info("💡 **Chasing Bias:** This ground shows a strong advantage for the team chasing (chase success rate is high). Teams winning the toss should prefer to **field** first.")
                    elif chase_success < 0.45:
                        st.info("💡 **Defending Bias:** Defending runs is easier on this surface. Teams winning the toss should prefer to **bat** first.")
                    else:
                        st.info("💡 **Neutral Pitch:** Toss has fairly low impact here; matches are evenly split between teams batting first and second.")
                        
        with tab2:
            st.markdown('<div class="subheader-custom">Compare Average Innings Scores Across Top Venues</div>', unsafe_allow_html=True)
            
            top_n = st.slider("Select number of top stadiums to compare:", min_value=5, max_value=25, value=10, step=1)
            fig_compare = va.plot_venue_average_scores(matches_df, deliveries_df, limit=top_n)
            st.plotly_chart(fig_compare, use_container_width=True)
            
    # ----------------------------------------------------
    # ADVANCED ANALYTICS PAGE
    # ----------------------------------------------------
    elif page == "📊 Advanced Analytics":
        render_header("ADVANCED ANALYTICS", "Deep dive into run distribution by phases, boundary ratios, and dot ball percentages")
        
        tab1, tab2 = st.tabs(["📈 Innings Phase Analysis", "🎯 Boundary & Dot Ball Ratios"])
        
        with tab1:
            st.markdown('<div class="subheader-custom">Runs and Wickets Distribution by Over Phases</div>', unsafe_allow_html=True)
            st.write(
                "We divide the cricket innings into three standard phases: "
                "**Powerplay (Overs 1-6)**, **Middle Overs (Overs 7-15)**, and **Death Overs (Overs 16-20)**."
            )
            
            # Create phase column
            dels_phase = deliveries_df.copy()
            dels_phase["phase"] = pd.cut(
                dels_phase["over"],
                bins=[0, 6, 15, 20],
                labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
            )
            
            # Group by phase to compute runs and balls
            phase_stats = dels_phase.groupby("phase").agg(
                Total_Runs=("total_runs", "sum"),
                Total_Balls=("wide_runs", lambda x: (x == 0).sum()), # Count balls excluding wides
                Wickets=("player_dismissed", lambda x: x.notna().sum())
            ).reset_index()
            
            phase_stats["Run Rate"] = ((phase_stats["Total_Runs"] / phase_stats["Total_Balls"]) * 6).round(2)
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                # Plotly runs distribution
                fig_pruns = px.bar(
                    phase_stats, x="phase", y="Total_Runs",
                    title="Total Runs Scored by Innings Phase",
                    color="Total_Runs", color_continuous_scale="Viridis",
                    text="Total_Runs"
                )
                fig_pruns.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_pruns, use_container_width=True)
            with col_p2:
                # Plotly Run rates
                fig_prr = px.bar(
                    phase_stats, x="phase", y="Run Rate",
                    title="Average Run Rate by Innings Phase",
                    color="Run Rate", color_continuous_scale="Turbo",
                    text="Run Rate"
                )
                fig_prr.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_prr, use_container_width=True)
                
            # Wickets by phase
            fig_pwick = px.pie(
                phase_stats, names="phase", values="Wickets",
                title="Wickets Fallen Distribution by Innings Phase",
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig_pwick.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_pwick, use_container_width=True)
            
        with tab2:
            st.markdown('<div class="subheader-custom">Boundary Ratios & Dot Ball Ratios by Teams</div>', unsafe_allow_html=True)
            
            # Select season to filter
            season_list = sorted(matches_df["season"].unique().tolist(), reverse=True)
            sel_season = st.selectbox("Select Season for Advanced Analytics:", ["All"] + [str(s) for s in season_list])
            
            # Process deliveries filtered by season
            if sel_season != "All":
                s_matches = matches_df[matches_df["season"] == int(sel_season)]["id"].tolist()
                del_filtered = deliveries_df[deliveries_df["match_id"].isin(s_matches)].copy()
            else:
                del_filtered = deliveries_df.copy()
                
            # Compute team-wise boundary and dot ball counts
            # Dot ball: batsman runs = 0 and extra runs = 0 (or no wide/noball runs)
            # Actually, standard dot ball is where total runs scored is 0
            del_filtered["is_dot"] = np.where(del_filtered["total_runs"] == 0, 1, 0)
            del_filtered["is_boundary"] = np.where(del_filtered["batsman_runs"].isin([4, 6]), 1, 0)
            
            team_adv = del_filtered.groupby("batting_team").agg(
                Total_Balls=("match_id", "size"),
                Dot_Balls=("is_dot", "sum"),
                Boundaries=("is_boundary", "sum"),
                Runs=("total_runs", "sum")
            ).reset_index()
            
            team_adv["Dot Ball %"] = ((team_adv["Dot_Balls"] / team_adv["Total_Balls"]) * 100).round(2)
            team_adv["Boundary % (Count)"] = ((team_adv["Boundaries"] / team_adv["Total_Balls"]) * 100).round(2)
            
            # Sort
            team_adv = team_adv.sort_values("Dot Ball %")
            
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                # Plotly Dot ball percentage
                fig_dots = px.bar(
                    team_adv, x="Dot Ball %", y="batting_team",
                    orientation="h",
                    title="Dot Ball Percentage by Batting Team (Lower is better)",
                    color="Dot Ball %", color_continuous_scale="Purp_r",
                    text="Dot Ball %"
                )
                fig_dots.update_layout(template="plotly_dark", height=450, yaxis_title=None, coloraxis_showscale=False)
                st.plotly_chart(fig_dots, use_container_width=True)
            with col_adv2:
                # Plotly boundary percentage
                team_adv_sort_b = team_adv.sort_values("Boundary % (Count)", ascending=False)
                fig_bounds = px.bar(
                    team_adv_sort_b, x="Boundary % (Count)", y="batting_team",
                    orientation="h",
                    title="Boundary Frequency % (Proportion of deliveries hit for 4/6)",
                    color="Boundary % (Count)", color_continuous_scale="Redor",
                    text="Boundary % (Count)"
                )
                fig_bounds.update_layout(template="plotly_dark", height=450, yaxis_title=None, coloraxis_showscale=False)
                st.plotly_chart(fig_bounds, use_container_width=True)
                
    # ----------------------------------------------------
    # MATCH WIN PREDICTOR PAGE
    # ----------------------------------------------------
    elif page == "🤖 Match Win Predictor":
        render_header("IPL WIN PREDICTOR (MACHINE LEARNING)", "Predict the win probability of chasing teams in live time using ML pipelines")
        
        # Train / Fetch model
        try:
            model_data = get_or_train_predictor(matches_df, deliveries_df)
            model_trained = True
        except Exception as e:
            st.error("Error loading or training the Win Predictor ML model.")
            st.exception(e)
            model_trained = False
            
        if model_trained:
            # Model accuracies info banner
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px; margin-bottom: 25px; font-size: 0.95rem;">
                    🤖 <b>Model Calibrations:</b> 
                    Logistic Regression Accuracy: <b>{model_data.get('lr_accuracy', 0.82)*100:.2f}%</b> | 
                    Random Forest Accuracy: <b>{model_data.get('rf_accuracy', 0.81)*100:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Predictor layout columns
            col_in, col_out = st.columns([1, 1])
            
            with col_in:
                st.markdown('<div class="subheader-custom" style="margin-top:0px;">Specify Current Match State</div>', unsafe_allow_html=True)
                
                # Fetch distinct list of teams and cities
                teams = model_data["teams"]
                cities = model_data["cities"]
                
                model_name = st.selectbox(
                    "Select Machine Learning Model:",
                    ["logistic_regression", "random_forest"],
                    format_func=lambda x: "Logistic Regression (Calibrated)" if x == "logistic_regression" else "Random Forest Classifier"
                )
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    batting_team = st.selectbox("Batting Team (Chasing):", teams, index=0)
                with col_t2:
                    # Filter bowling team to exclude batting team
                    bowling_teams = [t for t in teams if t != batting_team]
                    bowling_team = st.selectbox("Bowling Team (Defending):", bowling_teams, index=0)
                    
                city = st.selectbox("Match Host City:", cities, index=cities.index("Bangalore") if "Bangalore" in cities else 0)
                
                target_runs = st.number_input("Target Score to Chase:", min_value=1, max_value=300, value=160, step=1)
                
                col_score, col_wickets = st.columns(2)
                with col_score:
                    current_score = st.number_input("Current Score (Batting Team):", min_value=0, max_value=300, value=100, step=1)
                with col_wickets:
                    wickets_fallen = st.slider("Wickets Fallen:", min_value=0, max_value=9, value=3)
                    
                col_ov, col_bl = st.columns(2)
                with col_ov:
                    overs_completed = st.slider("Completed Overs Bowled:", min_value=0, max_value=19, value=12)
                with col_bl:
                    balls_in_over = st.slider("Balls Bowled in current over:", min_value=0, max_value=5, value=0)
                    
                # Combine overs and balls
                overs_input = float(overs_completed + balls_in_over / 10.0)
                
                # Validate inputs
                runs_left = target_runs - current_score
                balls_played = overs_completed * 6 + balls_in_over
                balls_left = 120 - balls_played
                
                inputs_valid = True
                if runs_left < 0:
                    st.error("Error: Current score cannot exceed the Target score.")
                    inputs_valid = False
                if current_score == 0 and balls_played > 0:
                    st.warning("Warning: Score is 0 runs but balls have been bowled. Ensure correct values.")
                if current_score > 0 and balls_played == 0:
                    st.error("Error: Cannot score runs with 0 balls delivered. Increase overs/balls completed.")
                    inputs_valid = False
                    
            with col_out:
                st.markdown('<div class="subheader-custom" style="margin-top:0px;">Win Probabilities</div>', unsafe_allow_html=True)
                
                if inputs_valid:
                    # Predict outcomes
                    bat_prob, bowl_prob = predict_match_outcome(
                        model_data=model_data,
                        model_name=model_name,
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        city=city,
                        target_runs=target_runs,
                        current_score=current_score,
                        wickets_fallen=wickets_fallen,
                        overs=overs_input
                    )
                    
                    # Display Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = round(bat_prob * 100, 1),
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{batting_team} Win Probability (%)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#3498db"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.2)'},
                                {'range': [40, 60], 'color': 'rgba(241, 196, 15, 0.2)'},
                                {'range': [60, 100], 'color': 'rgba(46, 204, 113, 0.2)'}
                            ],
                            'threshold': {
                                'line': {'color': "orange", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=280,
                        margin=dict(l=30, r=30, t=50, b=10)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Side-by-side percentage comparison bars
                    bat_pct = round(bat_prob * 100, 1)
                    bowl_pct = round(bowl_prob * 100, 1)
                    
                    st.markdown("##### Probabilities Split:")
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight:600; color:#3498db;">🏏 {batting_team} (Batting)</span>
                                <span style="font-weight:700; color:#3498db;">{bat_pct}%</span>
                            </div>
                            <div style="background-color: rgba(255,255,255,0.08); border-radius: 4px; height: 16px; width: 100%;">
                                <div style="background-color: #3498db; border-radius: 4px; height: 16px; width: {bat_pct}%;"></div>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight:600; color:#e74c3c;">🛡️ {bowling_team} (Bowling)</span>
                                <span style="font-weight:700; color:#e74c3c;">{bowl_pct}%</span>
                            </div>
                            <div style="background-color: rgba(255,255,255,0.08); border-radius: 4px; height: 16px; width: 100%;">
                                <div style="background-color: #e74c3c; border-radius: 4px; height: 16px; width: {bowl_pct}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Live Match Situation summary
                    st.markdown("##### Match Chase Context")
                    st.markdown(
                        f"""
                        - Required: **{runs_left}** runs needed off **{balls_left}** balls.
                        - Wickets remaining: **{10 - wickets_fallen}** / 10.
                        - Current Run Rate (CRR): **{current_score * 6 / balls_played if balls_played > 0 else 0:.2f}**
                        - Required Run Rate (RRR): **{runs_left * 6 / balls_left if balls_left > 0 else 0:.2f}**
                        """
                    )
                else:
                    st.info("Resolve input errors/warnings in the left panel to calculate win probabilities.")
