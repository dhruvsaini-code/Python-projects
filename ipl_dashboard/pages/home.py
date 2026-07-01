import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.ui import render_header, render_hero_banner, kpi_card
from config.settings import PLOTLY_CONFIG

def _render_home_kpi(filtered_matches: pd.DataFrame, filtered_deliveries: pd.DataFrame) -> None:
    """Renders the top KPI cards overview on the landing page."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        kpi_card("Filtered Matches", f"{len(filtered_matches)}", "kpi-green")
    with col2:
        kpi_card("Seasons Covered", f"{filtered_matches['season'].nunique()}", "kpi-orange")
    with col3:
        kpi_card("Host Venues", f"{filtered_matches['venue'].nunique()}", "kpi-purple")
    with col4:
        kpi_card("Total Runs Scored", f"{filtered_deliveries['total_runs'].sum():,}")
    with col5:
        kpi_card("Total Wickets Taken", f"{filtered_deliveries['player_dismissed'].notna().sum():,}")

def _render_overview_tab(filtered_matches: pd.DataFrame, plotly_template: str) -> None:
    """Renders the highlight metrics and match logs table in Overview tab."""
    col_s, col_t = st.columns([1, 2])
    with col_s:
        st.markdown('<div class="subheader-custom">IPL Dashboard Highlights</div>', unsafe_allow_html=True)
        st.write(
            "Welcome to the premium IPL Data Analytics Hub. Use the sidebar controls to filter the database "
            "globally. Your selections will apply directly to all subsequent analytics pages."
        )
        
        matches_per_season = filtered_matches["season"].value_counts().reset_index()
        matches_per_season.columns = ["Season", "Matches"]
        matches_per_season = matches_per_season.sort_values("Season")
        
        fig = px.bar(matches_per_season, x="Season", y="Matches", title="Matches Played by Season", color="Matches", color_continuous_scale="Agsunset")
        fig.update_layout(template=plotly_template, height=270, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
    with col_t:
        st.markdown('<div class="subheader-custom">Recent Matches & Outlines</div>', unsafe_allow_html=True)
        tbl_df = filtered_matches[["season", "date", "team1", "team2", "venue", "toss_winner", "winner", "result"]].copy()
        tbl_df.columns = [c.capitalize() for c in tbl_df.columns]
        
        st.dataframe(tbl_df.style.highlight_max(subset=["Winner"], color="rgba(46, 204, 113, 0.15)"), height=250, use_container_width=True)
        
        csv_data = tbl_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Matches List as CSV", data=csv_data, file_name="Filtered_IPL_Matches.csv", mime="text/csv")

def _render_comparison_tab(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders season-by-season analytics and line charts in Comparison tab."""
    st.markdown('<div class="subheader-custom">Season Comparison Dashboard</div>', unsafe_allow_html=True)
    
    season_data = []
    for s in sorted(matches_df["season"].unique()):
        s_matches = matches_df[matches_df["season"] == s]
        s_dels = deliveries_df[deliveries_df["match_id"].isin(s_matches["id"])]
        
        s_runs = s_dels["total_runs"].sum()
        season_data.append({
            "Season": str(s),
            "Matches": len(s_matches),
            "Avg Runs/Match": round(s_runs / len(s_matches), 1) if len(s_matches) > 0 else 0,
            "Total Sixes": len(s_dels[s_dels["batsman_runs"] == 6]),
            "Total Fours": len(s_dels[s_dels["batsman_runs"] == 4]),
            "Total Wickets": s_dels["player_dismissed"].notna().sum()
        })
        
    comp_df = pd.DataFrame(season_data)
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        fig1 = px.line(comp_df, x="Season", y="Avg Runs/Match", title="Average Runs Scored per Match by Season", markers=True)
        fig1.update_layout(template=plotly_template, height=320)
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
    with col_sc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=comp_df["Season"], y=comp_df["Total Sixes"], name="Sixes", marker_color="#f97316"))
        fig2.add_trace(go.Bar(x=comp_df["Season"], y=comp_df["Total Fours"], name="Fours", marker_color="#3b82f6"))
        fig2.update_layout(template=plotly_template, title="Boundary Trends over Seasons", barmode="stack", height=320)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
        
    st.dataframe(comp_df, use_container_width=True)

def show_home_page(filtered_matches: pd.DataFrame, filtered_deliveries: pd.DataFrame, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Home Page layout."""
    render_header("IPL DATA ANALYTICS & WIN PREDICTOR", "Interactive Analysis Dashboard & Machine Learning Match Outcome Predictor")
    render_hero_banner()
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    
    _render_home_kpi(filtered_matches, filtered_deliveries)
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    
    tab_overview, tab_comparison = st.tabs(["🏠 Overview", "📅 Season Comparison"])
    with tab_overview:
        _render_overview_tab(filtered_matches, plotly_template)
    with tab_comparison:
        _render_comparison_tab(matches_df, deliveries_df, plotly_template)
