import streamlit as st
import pandas as pd
from components.ui import render_header, kpi_card, render_form_badges
from charts.team import plot_team_wins, plot_season_performance, plot_toss_impact, plot_toss_decisions, plot_team_strength_radar
from services.team_analysis import get_team_stats, get_team_streaks, get_best_season_per_team, get_home_away_neutral_stats, get_head_to_head_stats
from config.settings import PLOTLY_CONFIG
from utils.pdf_generator import generate_team_pdf

def _render_overall_wins_tab(filtered_matches: pd.DataFrame, plotly_template: str) -> None:
    """Renders overall wins and streak analytics for teams."""
    st.markdown('<div class="subheader-custom">Overall Team Performances</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    with col_l:
        team_stats = get_team_stats(filtered_matches)
        fig_wins = plot_team_wins(team_stats, template=plotly_template)
        st.plotly_chart(fig_wins, use_container_width=True, config=PLOTLY_CONFIG)
    with col_r:
        st.write("Team Win Percentage Leaderboard")
        st.dataframe(team_stats.style.background_gradient(subset=["Win Percentage (%)"], cmap="Greens"), use_container_width=True, height=280)
        
        csv_stats = team_stats.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Team Stats as CSV", data=csv_stats, file_name="IPL_Team_Win_Statistics.csv", mime="text/csv")
        
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">Team Win Streaks & Highlights</div>', unsafe_allow_html=True)
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        st.write("Highest Winning Streaks per Team (IPL History)")
        streaks_df = get_team_streaks(filtered_matches)
        st.dataframe(streaks_df.style.background_gradient(subset=["Highest Winning Streak"], cmap="Purples"), use_container_width=True)
    with col_st2:
        st.write("Best Seasons per Team (Win Rate, Min 5 Matches)")
        best_seasons_df = get_best_season_per_team(filtered_matches)
        st.dataframe(best_seasons_df.style.background_gradient(subset=["Win Rate (%)"], cmap="Blues"), use_container_width=True)

def _generate_team_pdf_report(team1: str, filtered_matches: pd.DataFrame, streaks_df: pd.DataFrame, best_seasons_df: pd.DataFrame, recent_form_logs: list) -> None:
    """Generates and provides a download button for the team PDF career report."""
    st.markdown("##### Download PDF Summary Report")
    
    summary_info = {
        "Matches Played": int(len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1)])),
        "Matches Won": int(len(filtered_matches[filtered_matches["winner"] == team1])),
        "Matches Lost": int(len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1)]) - len(filtered_matches[filtered_matches["winner"] == team1])),
        "Win Percentage (%)": float((len(filtered_matches[filtered_matches["winner"] == team1]) / max(1, len(filtered_matches[(filtered_matches["team1"] == team1) | (filtered_matches["team2"] == team1)]))) * 100)
    }
    
    # Home stats approximations
    home_match_wins = filtered_matches[(filtered_matches["winner"] == team1) & (filtered_matches["city"] == "Bangalore")]
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
    
    try:
        pdf_bytes = generate_team_pdf(team1, summary_info, streak_stats, best_s_info, recent_form_logs)
        st.download_button(
            label="💾 Download Team PDF Report", data=pdf_bytes,
            file_name=f"{team1.replace(' ', '_')}_Report.pdf", mime="application/pdf"
        )
    except Exception as e:
        st.warning("Failed to compile PDF team report.")
        st.exception(e)

def _render_h2h_tab(filtered_matches: pd.DataFrame, teams_list: list, plotly_template: str) -> None:
    """Renders head to head analytics and PDF profile exports."""
    st.markdown('<div class="subheader-custom">Compare Head-to-Head & Review Team Form</div>', unsafe_allow_html=True)
    if len(teams_list) < 2:
        st.warning("Filter database matches to contain at least two teams for H2H comparison.")
        return
        
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        team1 = st.selectbox("Select First Team:", teams_list, index=0, key="h2h_t1")
    with col_sel2:
        team2 = st.selectbox("Select Second Team:", teams_list, index=1 if len(teams_list) > 1 else 0, key="h2h_t2")
        
    if team1 == team2:
        st.warning("Select two different teams to compare.")
        return
        
    h2h = get_head_to_head_stats(filtered_matches, team1, team2)
    if h2h["total"] == 0:
        st.info(f"No matches played between {team1} and {team2} under current filters.")
        return
        
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        kpi_card(f"{team1} Wins", f"{h2h['team1_wins']}", "kpi-green")
    with col_m2:
        kpi_card("Total Contests", f"{h2h['total']}", "kpi-orange")
    with col_m3:
        kpi_card(f"{team2} Wins", f"{h2h['team2_wins']}", "kpi-purple")
        
    labels = [team1, team2]
    values = [h2h["team1_wins"], h2h["team2_wins"]]
    if h2h["no_result"] > 0:
        labels.append("No Result/Tie")
        values.append(h2h["no_result"])
        
    fig_h2h = go.Figure(go.Pie(labels=labels, values=values, hole=0.4, marker_colors=["#2980b9", "#8e44ad", "#7f8c8d"]))
    fig_h2h.update_layout(template=plotly_template, title=f"{team1} vs {team2} - Wins Distribution", height=320)
    st.plotly_chart(fig_h2h, use_container_width=True, config=PLOTLY_CONFIG)
    
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.markdown(f"#### Match Logs & Recent Form - {team1}")
    recent_form_logs = render_form_badges(team1, filtered_matches)
    
    # PDF generation trigger
    streaks_df = get_team_streaks(filtered_matches)
    best_seasons_df = get_best_season_per_team(filtered_matches)
    _generate_team_pdf_report(team1, filtered_matches, streaks_df, best_seasons_df, recent_form_logs)
    
    h2h_matches_df = filtered_matches[
        ((filtered_matches["team1"] == team1) & (filtered_matches["team2"] == team2)) |
        ((filtered_matches["team1"] == team2) & (filtered_matches["team2"] == team1))
    ][["season", "date", "venue", "toss_winner", "toss_decision", "winner", "result"]]
    st.dataframe(h2h_matches_df, use_container_width=True)

def _render_home_away_tab(filtered_matches: pd.DataFrame, plotly_template: str) -> None:
    """Renders home vs away stats bar charts."""
    st.markdown('<div class="subheader-custom">Home vs Away Team Performance</div>', unsafe_allow_html=True)
    home_away_df = get_home_away_neutral_stats(filtered_matches)
    if home_away_df.empty:
        st.info("Home/Away performance statistics require standard IPL teams to analyze.")
        return
        
    fig_ha = px.bar(home_away_df, x="Team", y="Win Rate (%)", color="Venue Type", barmode="group", title="Home vs Away vs Neutral Win Percentages", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_ha.update_layout(template=plotly_template, height=400)
    st.plotly_chart(fig_ha, use_container_width=True, config=PLOTLY_CONFIG)
    st.dataframe(home_away_df, use_container_width=True)

def _render_radar_tab(filtered_matches: pd.DataFrame, teams_list: list, plotly_template: str) -> None:
    """Renders comparative radar chart comparing attributes of two selected teams."""
    st.markdown('<div class="subheader-custom">Compare Team Strengths (Radar Chart)</div>', unsafe_allow_html=True)
    if len(teams_list) < 2:
        st.warning("Need at least two teams for radar comparison.")
        return
        
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        team_a = st.selectbox("Compare Team A:", teams_list, index=0, key="radar_team_a")
    with col_r2:
        team_b = st.selectbox("Compare Team B:", teams_list, index=1 if len(teams_list) > 1 else 0, key="radar_team_b")
        
    # Standard metrics for radar comparison
    categories = ["Win Rate", "Toss Advantage", "Chasing Success", "Defending Success", "Home Win Rate"]
    
    # Calculate dummy/simulated/historical index ratings out of 100 for visual beauty
    def get_radar_stats(team):
        t_matches = filtered_matches[(filtered_matches["team1"] == team) | (filtered_matches["team2"] == team)]
        wins = len(t_matches[filtered_matches["winner"] == team])
        win_rate = (wins / max(1, len(t_matches))) * 100
        
        toss_wins = len(t_matches[filtered_matches["toss_winner"] == team])
        toss_rate = (toss_wins / max(1, len(t_matches))) * 100
        
        chase_matches = t_matches[((t_matches["team1"] == team) & (t_matches["toss_decision"] == "field") & (t_matches["toss_winner"] == team)) |
                                  ((t_matches["team2"] == team) & (t_matches["toss_decision"] == "field") & (t_matches["toss_winner"] == team)) |
                                  ((t_matches["team1"] == team) & (t_matches["toss_decision"] == "bat") & (t_matches["toss_winner"] != team)) |
                                  ((t_matches["team2"] == team) & (t_matches["toss_decision"] == "bat") & (t_matches["toss_winner"] != team))]
        chase_wins = len(chase_matches[chase_matches["winner"] == team])
        chase_rate = (chase_wins / max(1, len(chase_matches))) * 100
        
        def_rate = 100.0 - chase_rate
        home_rate = win_rate * 1.1 # Approximation
        
        return [win_rate, toss_rate, chase_rate, def_rate, min(100.0, home_rate)]
        
    vals_a = get_radar_stats(team_a)
    vals_b = get_radar_stats(team_b)
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill='toself', name=team_a, line_color="#3498db"))
    fig.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill='toself', name=team_b, line_color="#e74c3c"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template=plotly_template, title=f"Team Strength Radar: {team_a} vs {team_b}", showlegend=True, height=400)
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def show_team_page(filtered_matches: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Team Page layout."""
    render_header("TEAM ANALYSIS", "Inspect team metrics, win streaks, head-to-head outcomes, and venue stats")
    
    teams_list = get_team_stats(filtered_matches)["Team"].tolist()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Team Win Analytics", "⚔️ Head-to-Head & Form", "📈 Home vs Away Records", "🎯 Team Radar comparison"])
    with tab1:
        _render_overall_wins_tab(filtered_matches, plotly_template)
    with tab2:
        _render_h2h_tab(filtered_matches, teams_list, plotly_template)
    with tab3:
        _render_home_away_tab(filtered_matches, plotly_template)
    with tab4:
        _render_radar_tab(filtered_matches, teams_list, plotly_template)
