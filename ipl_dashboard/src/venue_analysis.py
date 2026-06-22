import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, Tuple, List

@st.cache_data(show_spinner=False)
def get_venue_list(matches_df: pd.DataFrame) -> List[str]:
    """
    Returns a list of venues ordered by the number of matches played there.
    """
    return matches_df["venue"].value_counts().index.tolist()

@st.cache_data(show_spinner=False)
def get_venue_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, venue_name: str) -> Dict[str, Any]:
    """
    Computes summary stats for a selected venue, including difficulty index and overs phase averages.
    """
    venue_matches = matches_df[matches_df["venue"] == venue_name]
    total_matches = len(venue_matches)
    
    if total_matches == 0:
        return {}
        
    # Calculate chasing vs defending wins
    chasing_wins = len(venue_matches[venue_matches["win_by_wickets"] > 0])
    defending_wins = len(venue_matches[venue_matches["win_by_runs"] > 0])
    no_results = total_matches - chasing_wins - defending_wins
    
    # Calculate average scores
    match_ids = venue_matches["id"].tolist()
    venue_dels = deliveries_df[deliveries_df["match_id"].isin(match_ids)].copy()
    
    inning_runs = venue_dels.groupby(["match_id", "inning"])["total_runs"].sum().reset_index()
    
    avg_1st_innings = inning_runs[inning_runs["inning"] == 1]["total_runs"].mean()
    avg_2nd_innings = inning_runs[inning_runs["inning"] == 2]["total_runs"].mean()
    
    avg_1st = round(avg_1st_innings, 1) if not pd.isna(avg_1st_innings) else 0.0
    avg_2nd = round(avg_2nd_innings, 1) if not pd.isna(avg_2nd_innings) else 0.0
    
    # Powerplay and Death overs average
    pp_runs = venue_dels[venue_dels["over"] <= 6].groupby(["match_id", "inning"])["total_runs"].sum().mean()
    death_runs = venue_dels[venue_dels["over"] >= 16].groupby(["match_id", "inning"])["total_runs"].sum().mean()
    
    avg_pp = round(pp_runs, 1) if not pd.isna(pp_runs) else 0.0
    avg_death = round(death_runs, 1) if not pd.isna(death_runs) else 0.0
    
    # Pitch Difficulty Index (Runs per Wicket: higher is easier for batsmen, lower is harder)
    total_venue_runs = venue_dels["total_runs"].sum()
    dismissed_series = venue_dels["player_dismissed"].astype(str).str.strip()
    total_venue_wickets = venue_dels[
        (~venue_dels["player_dismissed"].isna()) & 
        (~dismissed_series.isin(["", "0", "nan", "None", "none"]))
    ].shape[0]
    
    difficulty_index = round(total_venue_runs / total_venue_wickets, 2) if total_venue_wickets > 0 else 0.0
    
    # Highest and Lowest scores
    all_scores = (
        inning_runs.merge(venue_matches[["id", "team1", "team2"]], left_on="match_id", right_on="id")
    )
    
    highest_row = all_scores.sort_values("total_runs", ascending=False)
    lowest_row = all_scores.sort_values("total_runs", ascending=True)
    
    highest_score = 0
    highest_team = "N/A"
    lowest_score = 0
    lowest_team = "N/A"
    
    if not highest_row.empty:
        highest_score = int(highest_row.iloc[0]["total_runs"])
        h_match_id = highest_row.iloc[0]["match_id"]
        h_inning = highest_row.iloc[0]["inning"]
        h_del = venue_dels[(venue_dels["match_id"] == h_match_id) & (venue_dels["inning"] == h_inning)]
        if not h_del.empty:
            highest_team = h_del.iloc[0]["batting_team"]
            
    if not lowest_row.empty:
        lowest_score = int(lowest_row.iloc[0]["total_runs"])
        l_match_id = lowest_row.iloc[0]["match_id"]
        l_inning = lowest_row.iloc[0]["inning"]
        l_del = venue_dels[(venue_dels["match_id"] == l_match_id) & (venue_dels["inning"] == l_inning)]
        if not l_del.empty:
            lowest_team = l_del.iloc[0]["batting_team"]
            
    toss_wins = len(venue_matches[venue_matches["toss_winner"] == venue_matches["winner"]])
    toss_win_match_win_pct = round((toss_wins / total_matches) * 100, 2) if total_matches > 0 else 0.0
    
    return {
        "venue": venue_name,
        "total_matches": total_matches,
        "chasing_wins": chasing_wins,
        "defending_wins": defending_wins,
        "no_results": no_results,
        "avg_1st_innings": avg_1st,
        "avg_2nd_innings": avg_2nd,
        "avg_powerplay": avg_pp,
        "avg_death": avg_death,
        "difficulty_index": difficulty_index,
        "highest_score": highest_score,
        "highest_team": highest_team,
        "lowest_score": lowest_score,
        "lowest_team": lowest_team,
        "toss_win_match_win_pct": toss_win_match_win_pct
    }

@st.cache_data(show_spinner=False)
def get_all_venue_comparison(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comparative analytics for all venues hosting 5 or more matches.
    """
    venue_counts = matches_df["venue"].value_counts().reset_index()
    venue_counts.columns = ["venue", "matches_hosted"]
    
    top_venues = venue_counts[venue_counts["matches_hosted"] >= 5]["venue"].tolist()
    
    rows = []
    for venue in top_venues:
        v_matches = matches_df[matches_df["venue"] == venue]
        m_ids = v_matches["id"].tolist()
        
        # Chase vs Defend
        chasing_wins = len(v_matches[v_matches["win_by_wickets"] > 0])
        defending_wins = len(v_matches[v_matches["win_by_runs"] > 0])
        total_results = chasing_wins + defending_wins
        
        chase_win_pct = (chasing_wins / total_results * 100) if total_results > 0 else 0.0
        
        v_dels = deliveries_df[deliveries_df["match_id"].isin(m_ids)].copy()
        
        # Inning Averages
        inning_runs = v_dels.groupby(["match_id", "inning"])["total_runs"].sum().reset_index()
        avg_1st = inning_runs[inning_runs["inning"] == 1]["total_runs"].mean()
        avg_2nd = inning_runs[inning_runs["inning"] == 2]["total_runs"].mean()
        
        # Runs per wicket
        total_runs = v_dels["total_runs"].sum()
        dismissed_series = v_dels["player_dismissed"].astype(str).str.strip()
        total_wickets = v_dels[
            (~v_dels["player_dismissed"].isna()) & 
            (~dismissed_series.isin(["", "0", "nan", "None", "none"]))
        ].shape[0]
        runs_per_wicket = (total_runs / total_wickets) if total_wickets > 0 else 0.0
        
        # Powerplay/Death averages
        pp_runs = v_dels[v_dels["over"] <= 6].groupby(["match_id", "inning"])["total_runs"].sum().mean()
        death_runs = v_dels[v_dels["over"] >= 16].groupby(["match_id", "inning"])["total_runs"].sum().mean()
        
        rows.append({
            "Venue": venue,
            "Matches Hosted": len(v_matches),
            "Chasing Win Rate (%)": round(chase_win_pct, 2),
            "Pitch Difficulty (Runs/Wkt)": round(runs_per_wicket, 2),
            "Avg 1st Innings": round(avg_1st, 1) if not pd.isna(avg_1st) else 0.0,
            "Avg 2nd Innings": round(avg_2nd, 1) if not pd.isna(avg_2nd) else 0.0,
            "Avg Powerplay (1-6)": round(pp_runs, 1) if not pd.isna(pp_runs) else 0.0,
            "Avg Death (16-20)": round(death_runs, 1) if not pd.isna(death_runs) else 0.0
        })
        
    return pd.DataFrame(rows)

def plot_venue_chasing_vs_defending(stats: Dict[str, Any], template: str = "plotly_dark") -> go.Figure:
    """
    Plots a pie chart of chasing vs defending wins.
    """
    labels = ["Chasing Team Wins", "Defending Team Wins", "No Result"]
    values = [stats["chasing_wins"], stats["defending_wins"], stats["no_results"]]
    
    fig = px.pie(
        names=labels,
        values=values,
        title=f"Match Outcome Distribution - {stats['venue']}",
        color_discrete_sequence=["#2980b9", "#27ae60", "#7f8c8d"],
        hole=0.4
    )
    
    fig.update_layout(
        template=template,
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textinfo="value+percent+label")
    return fig

def plot_venue_average_scores(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, limit: int = 10, template: str = "plotly_dark") -> go.Figure:
    """
    Plots the average 1st and 2nd innings scores for the top venues.
    """
    comp_df = get_all_venue_comparison(matches_df, deliveries_df).head(limit)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df["Venue"],
        y=comp_df["Avg 1st Innings"],
        name="1st Innings Avg Score",
        marker_color="#e74c3c"
    ))
    fig.add_trace(go.Bar(
        x=comp_df["Venue"],
        y=comp_df["Avg 2nd Innings"],
        name="2nd Innings Avg Score",
        marker_color="#3498db"
    ))
    
    fig.update_layout(
        template=template,
        title=f"Average Innings Scores at Top {limit} Venues",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Average Score"),
        barmode="group",
        height=450,
        margin=dict(l=20, r=20, t=50, b=80)
    )
    return fig
