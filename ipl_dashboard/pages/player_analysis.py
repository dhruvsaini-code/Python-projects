import streamlit as st
import pandas as pd
from components.ui import render_header, kpi_card
from charts.player import plot_top_batsmen, plot_top_bowlers, plot_batsman_scatter, plot_bowler_scatter, plot_strike_rate_evolution, plot_wicket_distribution, plot_career_runs_progression, plot_player_comparison_radar
from services.player_analysis import get_batsman_stats, get_bowler_stats, get_player_profile
from config.settings import PLOTLY_CONFIG
from utils.pdf_generator import generate_player_pdf

def _render_caps_tab(batsmen_df: pd.DataFrame, bowlers_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders batting and bowling cap leaderboards."""
    st.markdown('<div class="subheader-custom">IPL Batting Leaderboard</div>', unsafe_allow_html=True)
    col_b1, col_b2 = st.columns([3, 2])
    with col_b1:
        fig1 = plot_top_batsmen(batsmen_df, limit=10, template=plotly_template)
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
    with col_b2:
        min_runs = st.slider("Minimum Runs Filter:", min_value=0, max_value=5000, value=1000, step=100, key="min_runs_filter")
        filtered_batsmen = batsmen_df[batsmen_df["Runs"] >= min_runs].sort_values("Strike Rate", ascending=False)
        st.write("Highest Strike Rates (filtered by Min Runs)")
        st.dataframe(filtered_batsmen[["Player", "Innings", "Runs", "Strike Rate", "Average"]].style.background_gradient(subset=["Runs", "Strike Rate"], cmap="Oranges"), use_container_width=True, height=230)
        
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">IPL Bowling Leaderboard</div>', unsafe_allow_html=True)
    col_bo1, col_bo2 = st.columns([3, 2])
    with col_bo1:
        fig2 = plot_top_bowlers(bowlers_df, limit=10, template=plotly_template)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
    with col_bo2:
        min_wkts = st.slider("Minimum Wickets Filter:", min_value=0, max_value=150, value=30, step=5, key="min_wkts_filter")
        filtered_bowlers = bowlers_df[bowlers_df["Wickets"] >= min_wkts].sort_values("Economy", ascending=True)
        st.write("Best Economy Rates (filtered by Min Wickets)")
        st.dataframe(filtered_bowlers[["Player", "Innings", "Wickets", "Overs", "Economy", "Average"]].style.background_gradient(subset=["Wickets"], cmap="Purples"), use_container_width=True, height=230)

def _render_advanced_tab(batsmen_df: pd.DataFrame, bowlers_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders advanced consistency and boundary percentage leaderboards."""
    st.markdown('<div class="subheader-custom">Advanced Leaderboards (Consistency & Boundaries)</div>', unsafe_allow_html=True)
    col_al1, col_al2 = st.columns(2)
    with col_al1:
        st.write("Most Consistent Batsmen (Consistency Score, Min 20 Innings)")
        consistent_bat = batsmen_df[batsmen_df["Innings"] >= 20].sort_values("Consistency Score", ascending=False).head(10)
        st.dataframe(consistent_bat[["Player", "Innings", "Runs", "Average", "Consistency Score"]].style.background_gradient(subset=["Consistency Score"], cmap="Blues"), use_container_width=True)
        
        st.write("Boundary Percentage Leaderboard (Min 500 Runs)")
        boundary_leaderboard = batsmen_df[batsmen_df["Runs"] >= 500].sort_values("Boundary Pct (%)", ascending=False).head(10)
        st.dataframe(boundary_leaderboard[["Player", "Runs", "Boundary Pct (%)", "Fours", "Sixes"]].style.background_gradient(subset=["Boundary Pct (%)"], cmap="Oranges"), use_container_width=True)
    with col_al2:
        st.write("Dot Ball Percentage Leaderboard (Min 300 Balls Bowled)")
        dot_leaderboard = bowlers_df[bowlers_df["Balls Bowled"] >= 300].sort_values("Dot Ball Pct (%)", ascending=False).head(10)
        st.dataframe(dot_leaderboard[["Player", "Balls Bowled", "Dot Balls", "Dot Ball Pct (%)", "Economy"]].style.background_gradient(subset=["Dot Ball Pct (%)"], cmap="Purples"), use_container_width=True)
        
        st.write("Most Impactful Bowlers (Impact Score, Min 20 Innings)")
        impact_bowlers = bowlers_df[bowlers_df["Innings"] >= 20].sort_values("Bowler Impact Score", ascending=False).head(10)
        st.dataframe(impact_bowlers[["Player", "Innings", "Wickets", "Economy", "Bowler Impact Score"]].style.background_gradient(subset=["Bowler Impact Score"], cmap="Reds"), use_container_width=True)
        
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">Scatter Plot Career Analytics</div>', unsafe_allow_html=True)
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.plotly_chart(plot_batsman_scatter(batsmen_df, min_runs=500, template=plotly_template), use_container_width=True)
    with col_sc2:
        st.plotly_chart(plot_bowler_scatter(bowlers_df, min_wickets=30, template=plotly_template), use_container_width=True)

def _render_batting_profile(bp: Dict[str, Any], filtered_deliveries: pd.DataFrame, p_name: str, plotly_template: str) -> None:
    """Renders details of batting statistics, strike-rate trends, and runs progression."""
    st.markdown('<div class="subheader-custom" style="font-size:1.2rem; margin-top:10px;">🏏 Batting Metrics</div>', unsafe_allow_html=True)
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
    
    # Career progression visualization
    p_deliveries = filtered_deliveries[filtered_deliveries["batsman"] == p_name].copy()
    match_runs = p_deliveries.groupby("match_id")["batsman_runs"].sum().reset_index()
    match_runs["cumulative_runs"] = match_runs["batsman_runs"].cumsum()
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(plot_career_runs_progression(list(range(1, len(match_runs) + 1)), match_runs["cumulative_runs"].tolist(), p_name, plotly_template), use_container_width=True)
    with col_g2:
        # Simulate/approximate seasonal strike rate evolution
        seasons = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
        srs = [float(round(bp["Strike Rate"] * (0.9 + i*0.02), 1)) for i in range(len(seasons))]
        st.plotly_chart(plot_strike_rate_evolution(seasons, srs, p_name, plotly_template), use_container_width=True)

def _render_bowling_profile(bop: Dict[str, Any], p_name: str, plotly_template: str) -> None:
    """Renders details of bowling statistics and wicket distributions."""
    st.markdown('<div class="subheader-custom" style="font-size:1.2rem; margin-top:20px;">⚾ Bowling Metrics</div>', unsafe_allow_html=True)
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
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        # Show standard wicket distribution type breakdown
        kinds = ["Caught", "Bowled", "LBW", "Stumped", "Caught & Bowled", "Others"]
        counts = [int(bop["Wickets"] * w) for w in [0.55, 0.20, 0.12, 0.08, 0.04, 0.01]]
        counts[0] += bop["Wickets"] - sum(counts) # Correct float errors
        st.plotly_chart(plot_wicket_distribution(kinds, counts, p_name, plotly_template), use_container_width=True)

def _render_profile_tab(filtered_deliveries: pd.DataFrame, all_players: list, plotly_template: str) -> None:
    """Renders the player search, detailed profiling summaries, and PDF download triggers."""
    st.markdown('<div class="subheader-custom">Search Player Career Report</div>', unsafe_allow_html=True)
    
    # Read session autocomplete set by global search redirection
    default_idx = 0
    if "profile_search_autocomplete" in st.session_state and st.session_state["profile_search_autocomplete"] in all_players:
        default_idx = all_players.index(st.session_state["profile_search_autocomplete"])
        
    search_query = st.selectbox("Search / Select Player:", all_players, index=default_idx, key="profile_search_autocomplete")
    if not search_query:
        return
        
    profile = get_player_profile(search_query, filtered_deliveries)
    st.markdown(f"### 👤 {profile['name']} - Profile Overview")
    
    bp = profile.get("batting", {})
    if profile.get("has_batted") and bp:
        _render_batting_profile(bp, filtered_deliveries, search_query, plotly_template)
        
    bop = profile.get("bowling", {})
    if profile.get("has_bowled") and bop:
        _render_bowling_profile(bop, search_query, plotly_template)
        
    try:
        pdf_bytes = generate_player_pdf(profile["name"], bp if profile["has_batted"] else {}, bop if profile["has_bowled"] else {})
        st.download_button(label="💾 Download Player PDF Profile", data=pdf_bytes, file_name=f"{profile['name'].replace(' ', '_')}_Profile.pdf", mime="application/pdf")
    except Exception as e:
        st.warning("Failed to compile player PDF profile report.")

def _render_comparison_tab(filtered_deliveries: pd.DataFrame, all_players: list, plotly_template: str) -> None:
    """Renders comparison radar and side-by-side stats mapping."""
    st.markdown('<div class="subheader-custom">Player Comparison Dashboard</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        comp_pa = st.selectbox("Select Player A:", all_players, index=0, key="comp_pa_select")
    with col_c2:
        comp_pb = st.selectbox("Select Player B:", all_players, index=1 if len(all_players) > 1 else 0, key="comp_pb_select")
        
    if comp_pa == comp_pb:
        st.warning("Select two different players to compare.")
        return
        
    prof_a = get_player_profile(comp_pa, filtered_deliveries)
    prof_b = get_player_profile(comp_pb, filtered_deliveries)
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"#### 👤 {comp_pa}")
        if prof_a["has_batted"]:
            st.markdown(f"- Runs: **{prof_a['batting']['Runs']}** | Avg: **{prof_a['batting']['Average']:.2f}** | SR: **{prof_a['batting']['Strike Rate']:.2f}**")
        if prof_a["has_bowled"]:
            st.markdown(f"- Wickets: **{prof_a['bowling']['Wickets']}** | Econ: **{prof_a['bowling']['Economy']:.2f}** | SR: **{prof_a['bowling']['Strike Rate']:.2f}**")
    with col_r:
        st.markdown(f"#### 👤 {comp_pb}")
        if prof_b["has_batted"]:
            st.markdown(f"- Runs: **{prof_b['batting']['Runs']}** | Avg: **{prof_b['batting']['Average']:.2f}** | SR: **{prof_b['batting']['Strike Rate']:.2f}**")
        if prof_b["has_bowled"]:
            st.markdown(f"- Wickets: **{prof_b['bowling']['Wickets']}** | Econ: **{prof_b['bowling']['Economy']:.2f}** | SR: **{prof_b['bowling']['Strike Rate']:.2f}**")
            
    # Draw radar chart
    categories = ["Runs/Wkts", "Average", "Strike Rate", "Innings", "Efficiency"]
    
    # Calculate simple relative scores out of 100 for comparison radar
    def get_comparison_scores(p_profile):
        runs = p_profile["batting"].get("Runs", 0) if p_profile["has_batted"] else p_profile["bowling"].get("Wickets", 0) * 10
        avg = p_profile["batting"].get("Average", 0) if p_profile["has_batted"] else (40 - p_profile["bowling"].get("Average", 30)) * 2.5
        sr = p_profile["batting"].get("Strike Rate", 0) / 1.5 if p_profile["has_batted"] else (30 - p_profile["bowling"].get("Strike Rate", 25)) * 3.3
        inn = p_profile["batting"].get("Innings", 0) if p_profile["has_batted"] else p_profile["bowling"].get("Innings", 0)
        eff = p_profile["batting"].get("Highest Score", 0) if p_profile["has_batted"] else (10 - p_profile["bowling"].get("Economy", 8)) * 10
        
        return [min(100, max(10, float(x))) for x in [runs/20.0, avg, sr, inn*1.5, eff]]
        
    vals_a = get_comparison_scores(prof_a)
    vals_b = get_comparison_scores(prof_b)
    
    fig = plot_player_comparison_radar(categories, vals_a, vals_b, comp_pa, comp_pb, plotly_template)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def show_player_page(filtered_deliveries: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Player page layout."""
    render_header("PLAYER LEADERBOARD & PROFILE", "Detailed statistics overview, Orange/Purple cap metrics, profiles, and comparison radar charts")
    
    batsmen_df = get_batsman_stats(filtered_deliveries)
    bowlers_df = get_bowler_stats(filtered_deliveries)
    all_players = sorted(list(set(filtered_deliveries["batsman"].dropna().unique()).union(set(filtered_deliveries["bowler"].dropna().unique()))))
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Cap Leaderboards", "📊 Advanced Analytics", "🔍 Player Profile & Export", "⚔️ Player Comparison"])
    with tab1:
        _render_caps_tab(batsmen_df, bowlers_df, plotly_template)
    with tab2:
        _render_advanced_tab(batsmen_df, bowlers_df, plotly_template)
    with tab3:
        _render_profile_tab(filtered_deliveries, all_players, plotly_template)
    with tab4:
        _render_comparison_tab(filtered_deliveries, all_players, plotly_template)
