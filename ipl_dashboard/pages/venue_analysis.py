import streamlit as st
import pandas as pd
from components.ui import render_header, kpi_card
from charts.venue import plot_venue_chasing_vs_defending, plot_venue_average_scores, plot_venue_phase_scores
from services.venue_analysis import get_venue_list, get_venue_stats, get_all_venue_comparison
from config.settings import PLOTLY_CONFIG

def _render_venue_specifics_tab(filtered_matches: pd.DataFrame, filtered_deliveries: pd.DataFrame, venues: list, plotly_template: str) -> None:
    """Renders detailed analytics for a single stadium."""
    st.markdown('<div class="subheader-custom">Analyze Individual Venue Details</div>', unsafe_allow_html=True)
    selected_venue = st.selectbox("Select Venue to Analyze:", venues, key="venue_specific_select")
    if not selected_venue:
        return
        
    v_stats = get_venue_stats(filtered_matches, filtered_deliveries, selected_venue)
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        kpi_card("Matches Hosted", f"{v_stats['total_matches']}", "kpi-green")
    with col_v2:
        kpi_card("Avg 1st Innings Score", f"{v_stats['avg_1st_innings']}", "kpi-orange")
    with col_v3:
        kpi_card("Toss Winner Win (%)", f"{v_stats['toss_win_match_win_pct']}%", "kpi-purple")
        
    col_vl, col_vr = st.columns(2)
    with col_vl:
        st.plotly_chart(plot_venue_chasing_vs_defending(v_stats, template=plotly_template), use_container_width=True)
    with col_vr:
        phases = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
        scores = [v_stats["avg_powerplay"], round(v_stats["avg_1st_innings"] - (v_stats["avg_powerplay"] + v_stats["avg_death"]), 1), v_stats["avg_death"]]
        st.plotly_chart(plot_venue_phase_scores(phases, scores, selected_venue, template=plotly_template), use_container_width=True)
        
    st.markdown(
        f"""
        <div class="kpi-card" style="margin-top: 15px;">
            <div style="color: var(--text-muted); font-size:0.9rem; text-transform:uppercase;">Pitch Difficulty Index</div>
            <div style="font-size: 1.8rem; font-weight:800; color:var(--text-color);">{v_stats['difficulty_index']} runs/wicket</div>
            <div style="color:var(--text-muted); font-size:0.85rem;">Higher indicates a flat batting-friendly wicket.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _render_comparative_tab(filtered_matches: pd.DataFrame, filtered_deliveries: pd.DataFrame, plotly_template: str) -> None:
    """Renders comparative venue bar charts and tabular indices."""
    st.markdown('<div class="subheader-custom">Compare Average Innings Scores Across Top Venues</div>', unsafe_allow_html=True)
    top_n = st.slider("Select number of top stadiums to compare:", min_value=5, max_value=25, value=10, step=1, key="top_venues_slider")
    
    comp_venue_df = get_all_venue_comparison(filtered_matches, filtered_deliveries)
    fig_compare = plot_venue_average_scores(comp_venue_df, limit=top_n, template=plotly_template)
    st.plotly_chart(fig_compare, use_container_width=True, config=PLOTLY_CONFIG)
    
    st.write("Venue Performance Index (filtered by minimum 5 matches hosted)")
    st.dataframe(comp_venue_df.style.background_gradient(subset=["Chasing Win Rate (%)"], cmap="Blues"), use_container_width=True)
    
    csv_venue_stats = comp_venue_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Venue Index Stats as CSV", data=csv_venue_stats, file_name="IPL_Venue_Performance_Index.csv", mime="text/csv")

def show_venue_page(filtered_matches: pd.DataFrame, filtered_deliveries: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Venue insights page layout."""
    render_header("VENUE STATS & ANALYTICS", "Understand stadium dynamics, runs by phases, and chasing biases")
    
    venues = get_venue_list(filtered_matches)
    tab1, tab2 = st.tabs(["🏟️ Venue Specifics", "📊 Comparative Venue Metrics"])
    with tab1:
        _render_venue_specifics_tab(filtered_matches, filtered_deliveries, venues, plotly_template)
    with tab2:
        _render_comparative_tab(filtered_matches, filtered_deliveries, plotly_template)
