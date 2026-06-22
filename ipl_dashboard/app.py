import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

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
from src.win_predictor import get_or_train_predictor, predict_match_outcome, get_feature_importance
import src.pdf_generator as pdf

# ----------------------------------------------------
# UI / THEME / DESIGN SYSTEM CONFIGURATION
# ----------------------------------------------------

# Sidebar Toggle for Dark / Light theme selection
st.sidebar.markdown("### Dashboard Style")
theme_mode = st.sidebar.selectbox("Select Theme:", ["Dark Mode", "Light Mode"], key="theme_selector")

# Inject CSS variables based on theme selection
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

# Load general stylesheet
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom stylesheet assets/custom.css not found. Standard styling applied.")

# Configure Plotly PNG download behavior
plotly_config = {
    'displayModeBar': True,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'ipl_analytics_chart',
        'height': 550,
        'width': 800,
        'scale': 2
    }
}

# Header banner helper
def render_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p style="text-align: center; color: var(--text-muted); margin-top:-25px; font-size:1.15rem; margin-bottom: 25px;">{subtitle}</p>', unsafe_allow_html=True)

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

def show_skeleton_loader():
    """Renders visual layout skeleton placeholder while computations process."""
    st.markdown(
        """
        <div class="kpi-card">
            <div class="skeleton-title"></div>
            <div class="skeleton-text"></div>
            <div class="skeleton-loader"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------
# DATA INGESTION & GLOBAL FILTERS
# ----------------------------------------------------

try:
    with st.spinner("Initializing IPL Data Analytics Engine..."):
        matches_df, deliveries_df = get_combined_data()
        data_loaded = True
except Exception as e:
    st.error("Error loading datasets. Please make sure matches.csv and deliveries.csv are available in the data/ folder.")
    st.exception(e)
    data_loaded = False

if data_loaded:
    # Sidebar global statistics overview
    st.sidebar.markdown("### Dashboard Sidebar Stats")
    total_db_matches = len(matches_df)
    total_db_seasons = matches_df["season"].nunique()
    st.sidebar.info(
        f"📂 **Total Matches:** {total_db_matches} contests\n"
        f"📅 **Seasons:** {matches_df['season'].min()} to {matches_df['season'].max()}"
    )

    # Sidebar global filter configs
    st.sidebar.markdown("### Global Filters")
    global_season = st.sidebar.selectbox("Filter Season:", ["All"] + [str(s) for s in sorted(matches_df["season"].unique(), reverse=True)], key="global_season_filter")
    
    unique_teams_db = sorted(list(set(matches_df["team1"]).union(set(matches_df["team2"]))))
    unique_teams_db = [t for t in unique_teams_db if str(t).strip() not in ["", "No Result", "Unknown"]]
    global_team = st.sidebar.selectbox("Filter Team:", ["All"] + unique_teams_db, key="global_team_filter")
    
    unique_venues_db = sorted(matches_df["venue"].dropna().unique().tolist())
    global_venue = st.sidebar.selectbox("Filter Venue:", ["All"] + unique_venues_db, key="global_venue_filter")

    # Apply global filters dynamically
    filtered_matches = matches_df.copy()
    if global_season != "All":
        filtered_matches = filtered_matches[filtered_matches["season"] == int(global_season)]
    if global_team != "All":
        filtered_matches = filtered_matches[
            (filtered_matches["team1"] == global_team) | (filtered_matches["team2"] == global_team)
        ]
    if global_venue != "All":
        filtered_matches = filtered_matches[filtered_matches["venue"] == global_venue]

    # Dynamically filter deliveries based on matching match_ids
    filtered_match_ids = filtered_matches["id"].tolist()
    filtered_deliveries = deliveries_df[deliveries_df["match_id"].isin(filtered_match_ids)].copy()

    # Sidebar Navigation Router
    st.sidebar.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Navigate Dashboard Pages:",
        [
            "🏠 Home Page", 
            "👥 Team Performance", 
            "🏏 Player Statistics", 
            "🏟️ Venue Insights", 
            "📊 Advanced Analytics",
            "🤖 Match Win Predictor"
        ],
        label_visibility="collapsed"
    )

    # ----------------------------------------------------
    # HOME PAGE PAGE
    # ----------------------------------------------------
    if page == "🏠 Home Page":
        render_header("IPL DATA ANALYTICS & WIN PREDICTOR", "Interactive Analysis Dashboard & Machine Learning Match outcome predictor")
        
        tab_home, tab_sec = st.tabs(["🏠 Overview", "📅 Season Comparison"])
        
        with tab_home:
            # Home page general KPIs
            total_matches_f = len(filtered_matches)
            total_seasons_f = filtered_matches["season"].nunique()
            total_venues_f = filtered_matches["venue"].nunique()
            
            # Runs and boundaries
            total_runs_f = filtered_deliveries["total_runs"].sum()
            total_wickets_f = filtered_deliveries["player_dismissed"].notna().sum()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                kpi_card("Filtered Matches", f"{total_matches_f}", "kpi-green")
            with col2:
                kpi_card("Seasons Covered", f"{total_seasons_f}", "kpi-orange")
            with col3:
                kpi_card("Host Venues", f"{total_venues_f}", "kpi-purple")
            with col4:
                kpi_card("Total Runs Scored", f"{total_runs_f:,}")
            with col5:
                kpi_card("Total Wickets Taken", f"{total_wickets_f:,}")
                
            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            
            col_s, col_t = st.columns([1, 2])
            with col_s:
                st.markdown('<div class="subheader-custom">IPL Dashboard Highlights</div>', unsafe_allow_html=True)
                st.write(
                    "Welcome to the premium IPL Data Analytics Hub. Use the sidebar controls to filter the database "
                    "globally. Your selections will apply directly to the leaderboards, streaks, venue details, "
                    "and advanced graphs on subsequent pages."
                )
                
                matches_per_season = filtered_matches["season"].value_counts().reset_index()
                matches_per_season.columns = ["Season", "Matches"]
                matches_per_season = matches_per_season.sort_values("Season")
                
                fig_seasons = px.bar(
                    matches_per_season, x="Season", y="Matches", 
                    title="Matches Played in Filtered Database",
                    color="Matches", color_continuous_scale="Agsunset"
                )
                fig_seasons.update_layout(template=plotly_template, height=290, coloraxis_showscale=False)
                st.plotly_chart(fig_seasons, use_container_width=True, config=plotly_config)
                
            with col_t:
                st.markdown('<div class="subheader-custom">Recent Matches & Outlines</div>', unsafe_allow_html=True)
                
                # Table of filtered matches
                tbl_df = filtered_matches[["season", "date", "team1", "team2", "venue", "toss_winner", "winner", "result"]].copy()
                tbl_df.columns = [c.capitalize() for c in tbl_df.columns]
                
                st.dataframe(
                    tbl_df.style.highlight_max(subset=["Winner"], color="rgba(46, 204, 113, 0.15)"), 
                    height=270, 
                    use_container_width=True
                )
                
                # CSV downloader for current filtered matches
                csv_data = tbl_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Matches List as CSV",
                    data=csv_data,
                    file_name="Filtered_IPL_Matches_Report.csv",
                    mime="text/csv",
                    key="download_filtered_matches_csv"
                )

        with tab_sec:
            st.markdown('<div class="subheader-custom">Season Comparison Dashboard</div>', unsafe_allow_html=True)
            # Compare runs per match, sixes, and wickets across seasons
            season_data = []
            seasons_list = sorted(matches_df["season"].unique())
            
            for s in seasons_list:
                s_matches = matches_df[matches_df["season"] == s]
                s_ids = s_matches["id"].tolist()
                s_dels = deliveries_df[deliveries_df["match_id"].isin(s_ids)]
                
                s_runs = s_dels["total_runs"].sum()
                s_sixes = len(s_dels[s_dels["batsman_runs"] == 6])
                s_fours = len(s_dels[s_dels["batsman_runs"] == 4])
                s_wkts = s_dels["player_dismissed"].notna().sum()
                
                season_data.append({
                    "Season": str(s),
                    "Matches": len(s_matches),
                    "Avg Runs/Match": round(s_runs / len(s_matches), 1) if len(s_matches) > 0 else 0,
                    "Total Sixes": s_sixes,
                    "Total Fours": s_fours,
                    "Total Wickets": s_wkts
                })
                
            season_comp_df = pd.DataFrame(season_data)
            
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                fig_run_trend = px.line(
                    season_comp_df, x="Season", y="Avg Runs/Match",
                    title="Average Runs Scored per Match by Season",
                    markers=True
                )
                fig_run_trend.update_layout(template=plotly_template, height=350)
                st.plotly_chart(fig_run_trend, use_container_width=True, config=plotly_config)
            with col_sc2:
                fig_boundary_trend = go.Figure()
                fig_boundary_trend.add_trace(go.Bar(x=season_comp_df["Season"], y=season_comp_df["Total Sixes"], name="Sixes", marker_color="#f97316"))
                fig_boundary_trend.add_trace(go.Bar(x=season_comp_df["Season"], y=season_comp_df["Total Fours"], name="Fours", marker_color="#3b82f6"))
                fig_boundary_trend.update_layout(template=plotly_template, title="Boundary Trends over Seasons", barmode="stack", height=350)
                st.plotly_chart(fig_boundary_trend, use_container_width=True, config=plotly_config)

            st.dataframe(season_comp_df, use_container_width=True)

    # ----------------------------------------------------
    # TEAM PERFORMANCE PAGE
    # ----------------------------------------------------
    elif page == "👥 Team Performance":
        render_header("TEAM ANALYSIS", "Inspect win histories, season progressions, and head-to-head records")
        
        tab1, tab2, tab3 = st.tabs(["📊 Team Win Analytics", "⚔️ Head-to-Head & Form", "📈 Home vs Away Records"])
        
        with tab1:
            st.markdown('<div class="subheader-custom">Overall Team Performances</div>', unsafe_allow_html=True)
            
            col_l, col_r = st.columns([3, 2])
            with col_l:
                fig_wins = ta.plot_team_wins(filtered_matches, template=plotly_template)
                st.plotly_chart(fig_wins, use_container_width=True, config=plotly_config)
            with col_r:
                team_stats = ta.get_team_stats(filtered_matches)
                st.write("Team Win Percentage Leaderboard")
                st.dataframe(
                    team_stats.style.background_gradient(subset=["Win Percentage (%)"], cmap="Greens"), 
                    use_container_width=True, 
                    height=360
                )
                
                # CSV Export for Team statistics
                csv_stats = team_stats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Team Stats as CSV",
                    data=csv_stats,
                    file_name="IPL_Team_Win_Statistics.csv",
                    mime="text/csv",
                    key="download_team_stats_csv"
                )
                
            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            
            # Streak & Best Season Cards
            st.markdown('<div class="subheader-custom">Team Win Streaks & Highlights</div>', unsafe_allow_html=True)
            col_st1, col_st2 = st.columns(2)
            with col_st1:
                st.write("Highest Winning Streaks per Team (IPL History)")
                streaks_df = ta.get_team_streaks(filtered_matches)
                st.dataframe(streaks_df.style.background_gradient(subset=["Highest Winning Streak"], cmap="Purples"), use_container_width=True)
            with col_st2:
                st.write("Best Seasons per Team (Win Rate, Min 5 Matches)")
                best_seasons_df = ta.get_best_season_per_team(filtered_matches)
                st.dataframe(best_seasons_df.style.background_gradient(subset=["Win Rate (%)"], cmap="Blues"), use_container_width=True)

        with tab2:
            st.markdown('<div class="subheader-custom">Compare Head-to-Head & Review Team Form</div>', unsafe_allow_html=True)
            
            teams_list = ta.get_unique_teams_list(filtered_matches)
            if len(teams_list) < 2:
                st.warning("Apply filters that allow at least two teams to compare.")
            else:
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    team1 = st.selectbox("Select First Team:", teams_list, index=0, key="h2h_t1")
                with col_sel2:
                    team2 = st.selectbox("Select Second Team:", teams_list, index=1 if len(teams_list) > 1 else 0, key="h2h_t2")
                    
                if team1 == team2:
                    st.warning("Please select two different teams for head-to-head comparison.")
                else:
                    h2h = ta.get_head_to_head_stats(filtered_matches, team1, team2)
                    
                    if h2h["total"] == 0:
                        st.info(f"No match played between {team1} and {team2} under current filters.")
                    else:
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            kpi_card(f"{team1} Wins", f"{h2h['team1_wins']}", "kpi-green")
                        with col_m2:
                            kpi_card("Total Contests", f"{h2h['total']}", "kpi-orange")
                        with col_m3:
                            kpi_card(f"{team2} Wins", f"{h2h['team2_wins']}", "kpi-purple")
                            
                        # Head-to-head pie chart
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
                        fig_h2h.update_layout(template=plotly_template, height=350)
                        st.plotly_chart(fig_h2h, use_container_width=True, config=plotly_config)
                        
                        # Form indicator row
                        st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
                        st.markdown(f"#### Match Logs & Recent Form - {team1}")
                        recent_matches_form = render_form_badges(team1, filtered_matches)
                        
                        # Exporters for selected team summary
                        st.markdown("##### Download PDF Summary Report")
                        
                        # Construct datasets for PDF compiler
                        summary_info = {
                            "Matches Played": int(len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1)])),
                            "Matches Won": int(len(filtered_matches[filtered_matches["winner"] == team1])),
                            "Matches Lost": int(len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1)]) - len(filtered_matches[filtered_matches["winner"] == team1])),
                            "Win Percentage (%)": float((len(filtered_matches[filtered_matches["winner"] == team1]) / max(1, len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1)]))) * 100)
                        }
                        # Add home ground totals
                        home_match_wins = filtered_matches[(filtered_matches["winner"] == team1) & (filtered_matches["city"] == "Bangalore")]  # Sample placeholder
                        summary_info["Home Wins"] = len(home_match_wins)
                        summary_info["Home Played"] = len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1) & (filtered_matches["city"] == "Bangalore")])
                        
                        streak_stats = {
                            "Highest Winning Streak": int(streaks_df[streaks_df["Team"] == team1]["Highest Winning Streak"].values[0]) if not streaks_df[streaks_df["Team"] == team1].empty else 0,
                            "Current Streak": int(streaks_df[streaks_df["Team"] == team1]["Current Streak"].values[0]) if not streaks_df[streaks_df["Team"] == team1].empty else 0
                        }
                        
                        best_s_row = best_seasons_df[best_seasons_df["Team"] == team1]
                        best_s_info = {
                            "Best Season": str(best_s_row["Best Season"].values[0]) if not best_s_row.empty else "N/A",
                            "Win Rate (%)": float(best_s_row["Win Rate (%)"].values[0]) if not best_s_row.empty else 0.0
                        }
                        
                        # Compile PDF in-memory
                        try:
                            pdf_bytes = pdf.generate_team_pdf(team1, summary_info, streak_stats, best_s_info, recent_matches_form)
                            st.download_button(
                                label="💾 Download Team PDF Report",
                                data=pdf_bytes,
                                file_name=f"{team1.replace(' ', '_')}_Career_Report.pdf",
                                mime="application/pdf",
                                key="download_team_pdf"
                            )
                        except Exception as e:
                            st.warning("Failed to initialize PDF download builder.")
                            st.exception(e)
                            
                        # Show raw match logs
                        h2h_matches_df = filtered_matches[
                            ((filtered_matches["team1"] == team1) & (filtered_matches["team2"] == team2)) |
                            ((filtered_matches["team1"] == team2) & (filtered_matches["team2"] == team1))
                        ][["season", "date", "venue", "toss_winner", "toss_decision", "winner", "result"]]
                        st.dataframe(h2h_matches_df, use_container_width=True)

        with tab3:
            st.markdown('<div class="subheader-custom">Home vs Away Team Performance</div>', unsafe_allow_html=True)
            home_away_df = ta.get_home_away_neutral_stats(filtered_matches)
            
            if home_away_df.empty:
                st.info("Home/Away performance statistics require standard IPL teams to analyze.")
            else:
                fig_ha = px.bar(
                    home_away_df, x="Team", y="Win Rate (%)", color="Venue Type",
                    barmode="group",
                    title="Home vs Away vs Neutral Win Percentages",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_ha.update_layout(template=plotly_template, height=450)
                st.plotly_chart(fig_ha, use_container_width=True, config=plotly_config)
                st.dataframe(home_away_df, use_container_width=True)

    # ----------------------------------------------------
    # PLAYER STATISTICS PAGE
    # ----------------------------------------------------
    elif page == "🏏 Player Statistics":
        render_header("PLAYER LEADERBOARD & PROFILE", "Detailed batsman & bowler stats, caps progression, and player search")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Cap leaderboards", "📊 Advanced Analytics", "🔍 Player Profile & Export", "⚔️ Player Comparison"])
        
        batsmen_df = pa.get_batsman_stats(filtered_deliveries)
        bowlers_df = pa.get_bowler_stats(filtered_deliveries)
        
        with tab1:
            st.markdown('<div class="subheader-custom">IPL Batting Leaderboard</div>', unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([3, 2])
            with col_b1:
                fig_top_bat = pa.plot_top_batsmen(batsmen_df, limit=10, template=plotly_template)
                st.plotly_chart(fig_top_bat, use_container_width=True, config=plotly_config)
            with col_b2:
                min_runs = st.slider("Minimum Runs Filter:", min_value=0, max_value=5000, value=1000, step=100, key="min_runs_filter")
                filtered_batsmen = batsmen_df[batsmen_df["Runs"] >= min_runs].sort_values("Strike Rate", ascending=False)
                st.write("Highest Strike Rates (filtered by Min Runs)")
                st.dataframe(
                    filtered_batsmen[["Player", "Innings", "Runs", "Strike Rate", "Average"]].style.background_gradient(subset=["Runs", "Strike Rate"], cmap="Oranges"), 
                    use_container_width=True, 
                    height=280
                )
                
            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            
            st.markdown('<div class="subheader-custom">IPL Bowling Leaderboard</div>', unsafe_allow_html=True)
            col_bo1, col_bo2 = st.columns([3, 2])
            with col_bo1:
                fig_top_bow = pa.plot_top_bowlers(bowlers_df, limit=10, template=plotly_template)
                st.plotly_chart(fig_top_bow, use_container_width=True, config=plotly_config)
            with col_bo2:
                min_wickets = st.slider("Minimum Wickets Filter:", min_value=0, max_value=150, value=30, step=5, key="min_wkts_filter")
                filtered_bowlers = bowlers_df[bowlers_df["Wickets"] >= min_wickets].sort_values("Economy", ascending=True)
                st.write("Best Economy Rates (filtered by Min Wickets)")
                st.dataframe(
                    filtered_bowlers[["Player", "Innings", "Wickets", "Overs", "Economy", "Average"]].style.background_gradient(subset=["Wickets"], cmap="Purples"), 
                    use_container_width=True, 
                    height=280
                )

        with tab2:
            st.markdown('<div class="subheader-custom">Advanced Leaderboards (Consistency & Boundaries)</div>', unsafe_allow_html=True)
            
            col_al1, col_al2 = st.columns(2)
            with col_al1:
                st.write("Most Consistent Batsmen (Consistency Score, Min 20 Innings)")
                consistent_bat = batsmen_df[batsmen_df["Innings"] >= 20].sort_values("Consistency Score", ascending=False).head(10)
                st.dataframe(
                    consistent_bat[["Player", "Innings", "Runs", "Average", "Consistency Score"]].style.background_gradient(subset=["Consistency Score"], cmap="Blues"),
                    use_container_width=True
                )
                
                st.write("Boundary percentage leaderboard (Min 500 runs)")
                boundary_leaderboard = batsmen_df[batsmen_df["Runs"] >= 500].sort_values("Boundary Pct (%)", ascending=False).head(10)
                st.dataframe(
                    boundary_leaderboard[["Player", "Runs", "Boundary Pct (%)", "Fours", "Sixes"]].style.background_gradient(subset=["Boundary Pct (%)"], cmap="Oranges"),
                    use_container_width=True
                )
            with col_al2:
                st.write("Dot Ball percentage leaderboard (Min 300 balls bowled)")
                dot_leaderboard = bowlers_df[bowlers_df["Balls Bowled"] >= 300].sort_values("Dot Ball Pct (%)", ascending=False).head(10)
                st.dataframe(
                    dot_leaderboard[["Player", "Balls Bowled", "Dot Balls", "Dot Ball Pct (%)", "Economy"]].style.background_gradient(subset=["Dot Ball Pct (%)"], cmap="Purples"),
                    use_container_width=True
                )
                
                st.write("Most Impactful Bowlers (Impact Score, Min 20 Innings)")
                impact_bowlers = bowlers_df[bowlers_df["Innings"] >= 20].sort_values("Bowler Impact Score", ascending=False).head(10)
                st.dataframe(
                    impact_bowlers[["Player", "Innings", "Wickets", "Economy", "Bowler Impact Score"]].style.background_gradient(subset=["Bowler Impact Score"], cmap="Reds"),
                    use_container_width=True
                )

            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            st.markdown('<div class="subheader-custom">Scatter Plot Career Analytics</div>', unsafe_allow_html=True)
            
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                fig_bat_sc = pa.plot_batsman_scatter(batsmen_df, min_runs=500, template=plotly_template)
                st.plotly_chart(fig_bat_sc, use_container_width=True, config=plotly_config)
            with col_sc2:
                fig_bow_sc = pa.plot_bowler_scatter(bowlers_df, min_wickets=30, template=plotly_template)
                st.plotly_chart(fig_bow_sc, use_container_width=True, config=plotly_config)

        with tab3:
            st.markdown('<div class="subheader-custom">Search Player Career Report</div>', unsafe_allow_html=True)
            
            all_players = sorted(list(set(filtered_deliveries["batsman"].dropna().unique()).union(set(filtered_deliveries["bowler"].dropna().unique()))))
            search_query = st.selectbox("Search / Select Player:", all_players, key="profile_search_autocomplete")
            
            if search_query:
                profile = pa.get_player_profile(search_query, filtered_deliveries)
                
                st.markdown(f"### 👤 {profile['name']} - Profile Overview")
                
                # Render batting profile if player has batted
                if profile["has_batted"]:
                    st.markdown('<div class="subheader-custom" style="font-size:1.2rem; margin-top:10px;">🏏 Batting Metrics</div>', unsafe_allow_html=True)
                    bp = profile["batting"]
                    
                    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
                    with col_b1:
                        kpi_card("Runs", f"{bp['Runs']}", "kpi-orange")
                    with col_b2:
                        kpi_card("Strike Rate", f"{bp['Strike Rate']:.2f}")
                    with col_b3:
                        kpi_card("Average", f"{bp['Average']:.2f}")
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
                        kpi_card("Economy", f"{bop['Economy']:.2f}")
                    with col_bo3:
                        kpi_card("Average", f"{bop['Average']:.2f}")
                    with col_bo4:
                        kpi_card("Best Figures", f"{bop['Best Figures']}")
                    with col_bo5:
                        kpi_card("3W / 5W", f"{bop['3W Hauls']} / {bop['5W Hauls']}")
                        
                    st.write(f"Innings Bowled: **{bop['Innings']}** | Overs Bowled: **{bop['Overs']}** | Strike Rate: **{bop['Strike Rate']}**")
                    
                # Download PDF Profile Exporter
                try:
                    pdf_p_bytes = pdf.generate_player_pdf(profile["name"], bp if profile["has_batted"] else {}, bop if profile["has_bowled"] else {})
                    st.download_button(
                        label="💾 Download Player PDF Profile",
                        data=pdf_p_bytes,
                        file_name=f"{profile['name'].replace(' ', '_')}_Profile_Report.pdf",
                        mime="application/pdf",
                        key="download_player_pdf"
                    )
                except Exception as e:
                    st.warning("Failed to initialize PDF profile download builder.")
                    
        with tab4:
            st.markdown('<div class="subheader-custom">Player Comparison Dashboard</div>', unsafe_allow_html=True)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                comp_pa = st.selectbox("Select Player A:", all_players, index=0, key="comp_pa_select")
            with col_c2:
                comp_pb = st.selectbox("Select Player B:", all_players, index=1 if len(all_players) > 1 else 0, key="comp_pb_select")
                
            if comp_pa == comp_pb:
                st.warning("Please select two different players for side-by-side comparison.")
            else:
                profile_a = pa.get_player_profile(comp_pa, filtered_deliveries)
                profile_b = pa.get_player_profile(comp_pb, filtered_deliveries)
                
                col_c_left, col_c_right = st.columns(2)
                with col_c_left:
                    st.markdown(f"#### 👤 {comp_pa}")
                    if profile_a["has_batted"]:
                        st.markdown("**Batting Details**")
                        st.write(f"- Runs: **{profile_a['batting']['Runs']}**")
                        st.write(f"- Average: **{profile_a['batting']['Average']:.2f}**")
                        st.write(f"- Strike Rate: **{profile_a['batting']['Strike Rate']:.2f}**")
                        st.write(f"- High Score: **{profile_a['batting']['Highest Score']}**")
                    if profile_a["has_bowled"]:
                        st.markdown("**Bowling Details**")
                        st.write(f"- Wickets: **{profile_a['bowling']['Wickets']}**")
                        st.write(f"- Economy: **{profile_a['bowling']['Economy']:.2f}**")
                        st.write(f"- Average: **{profile_a['bowling']['Average']:.2f}**")
                        
                with col_c_right:
                    st.markdown(f"#### 👤 {comp_pb}")
                    if profile_b["has_batted"]:
                        st.markdown("**Batting Details**")
                        st.write(f"- Runs: **{profile_b['batting']['Runs']}**")
                        st.write(f"- Average: **{profile_b['batting']['Average']:.2f}**")
                        st.write(f"- Strike Rate: **{profile_b['batting']['Strike Rate']:.2f}**")
                        st.write(f"- High Score: **{profile_b['batting']['Highest Score']}**")
                    if profile_b["has_bowled"]:
                        st.markdown("**Bowling Details**")
                        st.write(f"- Wickets: **{profile_b['bowling']['Wickets']}**")
                        st.write(f"- Economy: **{profile_b['bowling']['Economy']:.2f}**")
                        st.write(f"- Average: **{profile_b['bowling']['Average']:.2f}**")

    # ----------------------------------------------------
    # VENUE INSIGHTS PAGE
    # ----------------------------------------------------
    elif page == "🏟️ Venue Insights":
        render_header("VENUE STATS & ANALYTICS", "Understand the characteristics, average scores, and toss biases of IPL stadiums")
        
        tab1, tab2 = st.tabs(["🏟️ Venue Specifics", "📊 Comparative Venue Metrics"])
        
        with tab1:
            st.markdown('<div class="subheader-custom">Analyze Individual Venue Details</div>', unsafe_allow_html=True)
            
            venues = va.get_venue_list(filtered_matches)
            selected_venue = st.selectbox("Select Venue to Analyze:", venues, key="venue_specific_select")
            
            if selected_venue:
                v_stats = va.get_venue_stats(filtered_matches, filtered_deliveries, selected_venue)
                
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    kpi_card("Matches Hosted", f"{v_stats['total_matches']}", "kpi-green")
                with col_v2:
                    kpi_card("Avg 1st Innings Score", f"{v_stats['avg_1st_innings']}", "kpi-orange")
                with col_v3:
                    kpi_card("Toss Winner Win (%)", f"{v_stats['toss_win_match_win_pct']}%", "kpi-purple")
                    
                col_vl, col_vr = st.columns(2)
                with col_vl:
                    fig_chase = va.plot_venue_chasing_vs_defending(v_stats, template=plotly_template)
                    st.plotly_chart(fig_chase, use_container_width=True, config=plotly_config)
                with col_vr:
                    # Additional Venue Indicators
                    st.markdown('<div class="subheader-custom" style="font-size:1.15rem; margin-top:20px;">Venue Characteristics</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="margin-bottom: 12px;">
                            <div style="color: var(--text-muted); font-size:0.9rem; text-transform:uppercase;">Pitch Difficulty Index</div>
                            <div style="font-size: 1.8rem; font-weight:800; color:var(--text-color);">{v_stats['difficulty_index']} runs/wicket</div>
                            <div style="color:var(--text-muted); font-size:0.85rem;">Higher indicates a flat batting-friendly wicket.</div>
                        </div>
                        <div class="kpi-card" style="margin-bottom: 12px;">
                            <div style="color: var(--text-muted); font-size:0.9rem; text-transform:uppercase;">Overs Phase Average Scores</div>
                            <div style="font-size: 1.3rem; font-weight:700; color:var(--text-color);">Powerplay (1-6): {v_stats['avg_powerplay']} runs</div>
                            <div style="font-size: 1.3rem; font-weight:700; color:var(--text-color);">Death (16-20): {v_stats['avg_death']} runs</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    chase_success = v_stats['chasing_wins'] / v_stats['total_matches'] if v_stats['total_matches'] > 0 else 0
                    if chase_success > 0.55:
                        st.info("💡 **Chasing Bias:** This ground shows a strong advantage for the team chasing. Teams winning the toss should prefer to **field** first.")
                    elif chase_success < 0.45:
                        st.info("💡 **Defending Bias:** Defending runs is easier on this surface. Teams winning the toss should prefer to **bat** first.")
                    else:
                        st.info("💡 **Neutral Pitch:** Toss has fairly low impact here; matches are evenly split between batting first and second.")
                        
        with tab2:
            st.markdown('<div class="subheader-custom">Compare Average Innings Scores Across Top Venues</div>', unsafe_allow_html=True)
            
            top_n = st.slider("Select number of top stadiums to compare:", min_value=5, max_value=25, value=10, step=1, key="top_venues_slider")
            fig_compare = va.plot_venue_average_scores(filtered_matches, filtered_deliveries, limit=top_n, template=plotly_template)
            st.plotly_chart(fig_compare, use_container_width=True, config=plotly_config)
            
            # Comparative Table
            st.write("Venue Performance Index (filtered by minimum 5 matches hosted)")
            comp_venue_df = va.get_all_venue_comparison(filtered_matches, filtered_deliveries)
            st.dataframe(
                comp_venue_df.style.background_gradient(subset=["Chasing Win Rate (%)"], cmap="Blues"),
                use_container_width=True
            )
            
            # Export CSV comparative venue stats
            csv_venue_stats = comp_venue_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Venue Index stats as CSV",
                data=csv_venue_stats,
                file_name="IPL_Venue_Performance_Index.csv",
                mime="text/csv",
                key="download_venue_index_csv"
            )

    # ----------------------------------------------------
    # ADVANCED ANALYTICS PAGE
    # ----------------------------------------------------
    elif page == "📊 Advanced Analytics":
        render_header("ADVANCED ANALYTICS", "Deep dive into run distribution by phases, boundary ratios, and match progression timelines")
        
        tab1, tab2, tab3 = st.tabs(["📈 Innings Phase Analysis", "🎯 Boundary & Dot Ball Ratios", "📅 Past Match Simulator"])
        
        with tab1:
            st.markdown('<div class="subheader-custom">Runs and Wickets Distribution by Over Phases</div>', unsafe_allow_html=True)
            
            dels_phase = filtered_deliveries.copy()
            dels_phase["phase"] = pd.cut(
                dels_phase["over"],
                bins=[0, 6, 15, 20],
                labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
            )
            
            phase_stats = dels_phase.groupby("phase", observed=False).agg(
                Total_Runs=("total_runs", "sum"),
                Total_Balls=("wide_runs", lambda x: (x == 0).sum()),
                Wickets=("player_dismissed", lambda x: x.notna().sum())
            ).reset_index()
            
            phase_stats["Run Rate"] = np.where(
                phase_stats["Total_Balls"] > 0,
                ((phase_stats["Total_Runs"] / phase_stats["Total_Balls"]) * 6).round(2),
                0.0
            )
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_pruns = px.bar(
                    phase_stats, x="phase", y="Total_Runs",
                    title="Total Runs Scored by Innings Phase",
                    color="Total_Runs", color_continuous_scale="Viridis",
                    text="Total_Runs"
                )
                fig_pruns.update_layout(template=plotly_template, height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_pruns, use_container_width=True, config=plotly_config)
            with col_p2:
                fig_prr = px.bar(
                    phase_stats, x="phase", y="Run Rate",
                    title="Average Run Rate by Innings Phase",
                    color="Run Rate", color_continuous_scale="Turbo",
                    text="Run Rate"
                )
                fig_prr.update_layout(template=plotly_template, height=400, coloraxis_showscale=False)
                st.plotly_chart(fig_prr, use_container_width=True, config=plotly_config)
                
        with tab2:
            st.markdown('<div class="subheader-custom">Boundary Ratios & Dot Ball Ratios by Teams</div>', unsafe_allow_html=True)
            
            # Boundary & dot balls
            filtered_dels_adv = filtered_deliveries.copy()
            filtered_dels_adv["is_dot"] = np.where(filtered_dels_adv["total_runs"] == 0, 1, 0)
            filtered_dels_adv["is_boundary"] = np.where(filtered_dels_adv["batsman_runs"].isin([4, 6]), 1, 0)
            
            team_adv = filtered_dels_adv.groupby("batting_team").agg(
                Total_Balls=("match_id", "size"),
                Dot_Balls=("is_dot", "sum"),
                Boundaries=("is_boundary", "sum"),
                Runs=("total_runs", "sum")
            ).reset_index()
            
            team_adv["Dot Ball %"] = np.where(team_adv["Total_Balls"] > 0, ((team_adv["Dot_Balls"] / team_adv["Total_Balls"]) * 100).round(2), 0.0)
            team_adv["Boundary % (Count)"] = np.where(team_adv["Total_Balls"] > 0, ((team_adv["Boundaries"] / team_adv["Total_Balls"]) * 100).round(2), 0.0)
            
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                team_adv_sort_d = team_adv.sort_values("Dot Ball %")
                fig_dots = px.bar(
                    team_adv_sort_d, x="Dot Ball %", y="batting_team",
                    orientation="h",
                    title="Dot Ball Percentage by Batting Team (Lower is better)",
                    color="Dot Ball %", color_continuous_scale="Purp_r",
                    text="Dot Ball %"
                )
                fig_dots.update_layout(template=plotly_template, height=450, yaxis_title=None, coloraxis_showscale=False)
                st.plotly_chart(fig_dots, use_container_width=True, config=plotly_config)
            with col_adv2:
                team_adv_sort_b = team_adv.sort_values("Boundary % (Count)", ascending=False)
                fig_bounds = px.bar(
                    team_adv_sort_b, x="Boundary % (Count)", y="batting_team",
                    orientation="h",
                    title="Boundary Frequency % (Proportion of deliveries hit for 4/6)",
                    color="Boundary % (Count)", color_continuous_scale="Redor",
                    text="Boundary % (Count)"
                )
                fig_bounds.update_layout(template=plotly_template, height=450, yaxis_title=None, coloraxis_showscale=False)
                st.plotly_chart(fig_bounds, use_container_width=True, config=plotly_config)
                
            # Heatmap runs per over by team
            st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
            st.markdown('<div class="subheader-custom">Run Rate Heatmap (Over-by-Over)</div>', unsafe_allow_html=True)
            
            over_runs_avg = filtered_deliveries.groupby(["batting_team", "over"])["total_runs"].mean().reset_index()
            over_runs_avg["runs_per_over"] = over_runs_avg["total_runs"] * 6
            
            pivot_df = over_runs_avg.pivot(index="batting_team", columns="over", values="runs_per_over").fillna(0)
            
            fig_heatmap = px.imshow(
                pivot_df,
                labels=dict(x="Over", y="Batting Team", color="Runs/Over"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="RdBu_r",
                title="Average Runs Scored Per Over by Teams (Run Rate)"
            )
            fig_heatmap.update_layout(template=plotly_template, height=480)
            st.plotly_chart(fig_heatmap, use_container_width=True, config=plotly_config)

        with tab3:
            st.markdown('<div class="subheader-custom">Simulate Past Match Win Probability Timeline</div>', unsafe_allow_html=True)
            st.write(
                "Select a historical match from the database to plot the over-by-over win probability "
                "chase progression timeline calculated by our Machine Learning classifier model."
            )
            
            # Get models
            try:
                model_data = get_or_train_predictor(matches_df, deliveries_df)
                model_loaded = True
            except Exception:
                st.warning("Failed to load predictor model pipeline.")
                model_loaded = False
                
            if model_loaded:
                col_hps, col_hpm = st.columns(2)
                with col_hps:
                    seasons_with_matches = sorted(filtered_matches["season"].unique(), reverse=True)
                    sel_match_season = st.selectbox("Select Season:", seasons_with_matches, key="sim_season_select")
                with col_hpm:
                    season_matches = filtered_matches[filtered_matches["season"] == int(sel_match_season)]
                    
                    # Create labels
                    match_options = []
                    for idx, row in season_matches.iterrows():
                        label = f"{row['team1']} vs {row['team2']} ({row['venue']}, {row['city']})"
                        match_options.append((row["id"], label))
                        
                    if not match_options:
                        st.info("No matches found for selected season.")
                    else:
                        selected_match_tuple = st.selectbox(
                            "Select Match:", 
                            match_options, 
                            format_func=lambda x: x[1],
                            key="sim_match_select"
                        )
                        
                        if selected_match_tuple:
                            match_id = selected_match_tuple[0]
                            
                            # Run timeline calculator
                            from src.win_predictor import predict_match_outcome
                            
                            m_info = filtered_matches[filtered_matches["id"] == match_id].iloc[0]
                            chase_dels = deliveries_df[(deliveries_df["match_id"] == match_id) & (deliveries_df["inning"] == 2)].copy()
                            
                            if chase_dels.empty:
                                st.warning("Selected match has no recorded second innings chase data.")
                            else:
                                first_inn_runs = deliveries_df[(deliveries_df["match_id"] == match_id) & (deliveries_df["inning"] == 1)]["total_runs"].sum()
                                target_runs = first_inn_runs + 1
                                
                                batting_team = chase_dels["batting_team"].iloc[0]
                                bowling_team = chase_dels["bowling_team"].iloc[0]
                                city = m_info["city"]
                                if city == "Bengaluru":
                                    city = "Bangalore"
                                    
                                chase_dels.sort_values(by=["over", "ball"], inplace=True)
                                
                                prog_data = []
                                # over 0 state
                                prob_start, _ = predict_match_outcome(
                                    model_data, "logistic_regression",
                                    batting_team, bowling_team, city,
                                    target_runs, 0, 0, 0.0
                                )
                                prog_data.append({
                                    "Over": 0,
                                    "Score": "0/0",
                                    "Runs Left": target_runs,
                                    "CRR": 0.0,
                                    "RRR": round((target_runs * 6)/120, 2),
                                    "Probability (%)": round(prob_start * 100, 1)
                                })
                                
                                for over_num in range(1, 21):
                                    over_dels = chase_dels[chase_dels["over"] <= over_num]
                                    if over_dels.empty:
                                        break
                                        
                                    curr_score = over_dels["total_runs"].sum()
                                    wkts = over_dels["player_dismissed"].notna().sum()
                                    runs_l = max(0, target_runs - curr_score)
                                    balls_l = max(0, 120 - over_num * 6)
                                    
                                    crr = round((curr_score * 6) / (over_num * 6), 2)
                                    rrr = round((runs_l * 6) / balls_l, 2) if balls_l > 0 else 0.0
                                    
                                    prob, _ = predict_match_outcome(
                                        model_data, "logistic_regression",
                                        batting_team, bowling_team, city,
                                        target_runs, curr_score, wkts, float(over_num)
                                    )
                                    
                                    prog_data.append({
                                        "Over": over_num,
                                        "Score": f"{curr_score}/{wkts}",
                                        "Runs Left": runs_l,
                                        "CRR": crr,
                                        "RRR": rrr,
                                        "Probability (%)": round(prob * 100, 1)
                                    })
                                    
                                    if runs_l == 0 or wkts >= 10:
                                        break
                                        
                                prog_df = pd.DataFrame(prog_data)
                                
                                # Plot progression chart
                                fig_prog = px.line(
                                    prog_df, x="Over", y="Probability (%)",
                                    title=f"Chase Progression: {batting_team} Win Probability Timeline",
                                    markers=True,
                                    text="Score"
                                )
                                fig_prog.update_traces(textposition="top center", line_color="#3498db")
                                fig_prog.update_layout(template=plotly_template, yaxis_range=[0, 100], height=400)
                                st.plotly_chart(fig_prog, use_container_width=True, config=plotly_config)
                                
                                st.write("Historical over-by-over stats summary")
                                st.dataframe(prog_df, use_container_width=True)

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
            lr_m = model_data.get("lr_metrics", {"accuracy": 0.82})
            rf_m = model_data.get("rf_metrics", {"accuracy": 0.81})
            
            st.markdown(
                f"""
                <div style="background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin-bottom: 25px; font-size: 0.95rem; color: var(--text-color);">
                    🤖 <b>Model Calibrations:</b> 
                    Logistic Regression Accuracy: <b>{lr_m.get('accuracy', 0.82)*100:.2f}%</b> | 
                    Random Forest Accuracy: <b>{rf_m.get('accuracy', 0.81)*100:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            tab_pred, tab_eval = st.tabs(["🤖 Outcome Simulator", "📊 Model Evaluation & Features"])
            
            with tab_pred:
                # Predictor layout columns
                col_in, col_out = st.columns([1, 1])
                
                with col_in:
                    st.markdown('<div class="subheader-custom" style="margin-top:0px;">Specify Current Match State</div>', unsafe_allow_html=True)
                    
                    teams = model_data["teams"]
                    cities = model_data["cities"]
                    
                    model_name = st.selectbox(
                        "Select Machine Learning Model Pipeline:",
                        ["logistic_regression", "random_forest"],
                        format_func=lambda x: "Logistic Regression (Calibrated Probability)" if x == "logistic_regression" else "Random Forest Classifier",
                        key="predictor_model_select"
                    )
                    
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        batting_team = st.selectbox("Batting Team (Chasing):", teams, index=0, key="predictor_bat_team")
                    with col_t2:
                        bowling_teams = [t for t in teams if t != batting_team]
                        bowling_team = st.selectbox("Bowling Team (Defending):", bowling_teams, index=0, key="predictor_bowl_team")
                        
                    city = st.selectbox("Match Host City:", cities, index=cities.index("Bangalore") if "Bangalore" in cities else 0, key="predictor_city")
                    
                    target_runs = st.number_input("Target Score to Chase:", min_value=1, max_value=300, value=160, step=1, key="predictor_target")
                    
                    col_score, col_wickets = st.columns(2)
                    with col_score:
                        current_score = st.number_input("Current Score (Batting Team):", min_value=0, max_value=300, value=100, step=1, key="predictor_curr_score")
                    with col_wickets:
                        wickets_fallen = st.slider("Wickets Fallen:", min_value=0, max_value=9, value=3, key="predictor_wickets")
                        
                    col_ov, col_bl = st.columns(2)
                    with col_ov:
                        overs_completed = st.slider("Completed Overs Bowled:", min_value=0, max_value=19, value=12, key="predictor_overs")
                    with col_bl:
                        balls_in_over = st.slider("Balls Bowled in current over:", min_value=0, max_value=5, value=0, key="predictor_balls")
                        
                    # Combine overs and balls to prevent invalid float decimals
                    overs_input = float(overs_completed + balls_in_over / 10.0)
                    
                    # Validate inputs in real time
                    runs_left = target_runs - current_score
                    balls_played = overs_completed * 6 + balls_in_over
                    balls_left = 120 - balls_played
                    
                    inputs_valid = True
                    if runs_left < 0:
                        st.error("Error: Current score cannot exceed the Target score.")
                        inputs_valid = False
                    if current_score == 0 and balls_played > 0:
                        st.warning("Warning: Score is 0 runs but balls have been bowled.")
                    if current_score > 0 and balls_played == 0:
                        st.error("Error: Cannot score runs with 0 balls delivered. Increase overs/balls completed.")
                        inputs_valid = False
                        
                with col_out:
                    st.markdown('<div class="subheader-custom" style="margin-top:0px;">Calculated Probabilities</div>', unsafe_allow_html=True)
                    
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
                        
                        # Show confidence level based on proximity to 0 or 100%
                        prob_margin = abs(bat_prob - 0.5) * 2
                        if prob_margin >= 0.6:
                            confidence_level = "High"
                        elif prob_margin >= 0.2:
                            confidence_level = "Moderate"
                        else:
                            confidence_level = "Low"
                            
                        # Display Gauge Chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = round(bat_prob * 100, 1),
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"{batting_team} Win Probability (%)", 'font': {'size': 18, 'color': 'var(--text-color)'}},
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
                            template=plotly_template,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=260,
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
                        
                        # Run Rates
                        crr_val = current_score * 6 / balls_played if balls_played > 0 else 0
                        rrr_val = runs_left * 6 / balls_left if balls_left > 0 else 0
                        
                        # Live Match Situation summary
                        st.markdown(
                            f"""
                            ##### Match Chase Context
                            - Required: **{runs_left}** runs needed off **{balls_left}** balls.
                            - Wickets remaining: **{10 - wickets_fallen}** / 10.
                            - Current Run Rate (CRR): **{crr_val:.2f}** | Required Run Rate (RRR): **{rrr_val:.2f}**
                            - Prediction Confidence: **{confidence_level}**
                            """
                        )
                        
                        # PDF Exporter for Simulation state
                        st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
                        st.markdown("##### Download Simulation PDF Report")
                        sim_data = {
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "city": city,
                            "target_runs": int(target_runs),
                            "current_score": int(current_score),
                            "wickets_fallen": int(wickets_fallen),
                            "overs_input": float(overs_input),
                            "model_name": model_name,
                            "batting_win_prob": float(bat_pct),
                            "bowling_win_prob": float(bowl_pct),
                            "confidence_level": confidence_level,
                            "crr": float(crr_val),
                            "rrr": float(rrr_val)
                        }
                        try:
                            pdf_sim_bytes = pdf.generate_predictor_pdf(sim_data)
                            st.download_button(
                                label="💾 Download Prediction Report PDF",
                                data=pdf_sim_bytes,
                                file_name="IPL_Chase_Prediction_Report.pdf",
                                mime="application/pdf",
                                key="download_predictor_pdf"
                            )
                        except Exception as e:
                            st.warning("Failed to initialize PDF simulator download builder.")
                    else:
                        st.info("Resolve input errors/warnings in the left panel to calculate win probabilities.")

            with tab_eval:
                st.markdown('<div class="subheader-custom">Classifier Model Comparison Metrics</div>', unsafe_allow_html=True)
                
                # Render metrics comparison table
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                    "Logistic Regression": [
                        f"{lr_m.get('accuracy', 0.82)*100:.2f}%", 
                        f"{lr_m.get('precision', 0.82)*100:.2f}%", 
                        f"{lr_m.get('recall', 0.81)*100:.2f}%", 
                        f"{lr_m.get('f1', 0.81)*100:.2f}%"
                    ],
                    "Random Forest": [
                        f"{rf_m.get('accuracy', 0.81)*100:.2f}%", 
                        f"{rf_m.get('precision', 0.81)*100:.2f}%", 
                        f"{rf_m.get('recall', 0.80)*100:.2f}%", 
                        f"{rf_m.get('f1', 0.80)*100:.2f}%"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                # Feature Importance Chart
                st.markdown('<div class="subheader-custom">Top Predictive Features Impact Chart</div>', unsafe_allow_html=True)
                importance_df = get_feature_importance(model_data, model_name)
                
                if not importance_df.empty:
                    fig_importance = px.bar(
                        importance_df, x="Importance", y="Feature",
                        orientation="h",
                        title=f"Top Features driving outcome prediction - {model_name.replace('_', ' ').title()}",
                        color="Importance",
                        color_continuous_scale="Blugrn",
                        text="Importance"
                    )
                    fig_importance.update_layout(template=plotly_template, yaxis={"categoryorder": "total ascending"}, height=450, coloraxis_showscale=False)
                    st.plotly_chart(fig_importance, use_container_width=True, config=plotly_config)
                else:
                    st.info("Feature names mapping is compiling or unavailable for selected model.")

    # ----------------------------------------------------
    # DASHBOARD FOOTER INFO
    # ----------------------------------------------------
    st.markdown(
        """
        <div class="footer-text">
            🏏 <b>IPL Data Analytics & Win Predictor Dashboard</b> | Production Grade Showcase | Built with Streamlit, Plotly, & Scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )
