import pandas as pd
import streamlit as st
from typing import Dict, Any, List

def render_header(title: str, subtitle: str = "") -> None:
    """Renders a beautiful premium landing header banner."""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<p style="text-align: center; color: var(--text-muted); margin-top:-25px; '
            f'font-size:1.15rem; margin-bottom: 25px;">{subtitle}</p>',
            unsafe_allow_html=True
        )

def kpi_card(title: str, value: str, style_class: str = "") -> None:
    """Helper function to create premium glass-styled KPI cards."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value {style_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_skeleton_loader() -> None:
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

def render_footer() -> None:
    """Renders dashboard footer branding details."""
    st.markdown(
        """
        <div class="footer-text">
            🏏 <b>IPL Data Analytics & Win Predictor Dashboard</b> | Production Grade Showcase | Built with Streamlit, Plotly, & Scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )

def render_hero_banner() -> None:
    """Renders a stunning hero introduction card with a subtle breathing pulse glow."""
    st.markdown(
        """
        <div class="kpi-card" style="animation: pulseGlow 4s infinite; text-align: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);">
            <h2 style="font-size: 2.2rem; font-weight: 800; margin: 0; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🏏 Elevate Your Cricket Analytics
            </h2>
            <p style="color: var(--text-muted); font-size: 1.15rem; max-width: 700px; margin: 15px auto 0 auto; line-height: 1.6;">
                Welcome to the next-generation IPL Data Engine. Dive deep into player profiles, simulate game outcomes, compare machine learning algorithms, and unlock premium sports intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_form_badges(team: str, matches_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Renders chronological HTML form badges (W, L, NR) in Streamlit.
    Returns the recent match logs for PDF export compilation.
    """
    team_matches = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)].copy()
    if team_matches.empty:
        st.write("No matches played by team under these filters.")
        return []
        
    team_matches["date_parsed"] = pd.to_datetime(team_matches["date"], errors="coerce")
    recent = team_matches.sort_values("date_parsed", ascending=False).head(10)
    
    badge_html = ""
    recent_list = []
    
    # Chronological: oldest to newest (left to right)
    for _, row in recent.iloc[::-1].iterrows():
        winner = row["winner"]
        if winner == team:
            badge_html += '<span class="badge-w">W</span>'
        elif winner in ["No Result", "Unknown"] or pd.isna(winner):
            badge_html += '<span class="badge-nr">NR</span>'
        else:
            badge_html += '<span class="badge-l">L</span>'
            
        opp = row["team2"] if row["team1"] == team else row["team1"]
        margin = row["win_by_runs"] if row["win_by_runs"] > 0 else row["win_by_wickets"]
        margin_type = "runs" if row["win_by_runs"] > 0 else "wickets"
        result_str = f"Won by {margin} {margin_type}" if winner == team else (f"Lost by {margin} {margin_type}" if winner != "No Result" else "No Result")
        
        recent_list.append({
            "date": str(row["date"]), "opponent": opp, "venue": str(row["venue"]),
            "city": str(row["city"]), "toss_winner": str(row["toss_winner"]),
            "toss_decision": str(row["toss_decision"]), "winner": str(winner), "result": result_str
        })
        
    st.markdown(f'<div style="margin: 15px 0; font-size: 1.1rem;"><b>Recent Form:</b> {badge_html}</div>', unsafe_allow_html=True)
    return recent_list
