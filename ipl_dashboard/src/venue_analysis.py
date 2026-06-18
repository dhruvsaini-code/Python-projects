import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

def get_venue_list(matches_df: pd.DataFrame) -> list:
    """
    Returns a list of venues ordered by the number of matches played there.
    """
    return matches_df["venue"].value_counts().index.tolist()

def get_venue_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, venue_name: str) -> Dict[str, Any]:
    """
    Computes summary stats for a selected venue.
    """
    venue_matches = matches_df[matches_df["venue"] == venue_name]
    total_matches = len(venue_matches)
    
    if total_matches == 0:
        return {}
        
    # Calculate chasing vs defending wins
    # win_by_wickets > 0 means chasing team won
    # win_by_runs > 0 means defending team won
    chasing_wins = len(venue_matches[venue_matches["win_by_wickets"] > 0])
    defending_wins = len(venue_matches[venue_matches["win_by_runs"] > 0])
    no_results = total_matches - chasing_wins - defending_wins
    
    # Calculate average scores
    match_ids = venue_matches["id"].tolist()
    venue_dels = deliveries_df[deliveries_df["match_id"].isin(match_ids)]
    
    inning_runs = venue_dels.groupby(["match_id", "inning"])["total_runs"].sum().reset_index()
    
    avg_1st_innings = inning_runs[inning_runs["inning"] == 1]["total_runs"].mean()
    avg_2nd_innings = inning_runs[inning_runs["inning"] == 2]["total_runs"].mean()
    
    avg_1st = round(avg_1st_innings, 1) if not pd.isna(avg_1st_innings) else 0.0
    avg_2nd = round(avg_2nd_innings, 1) if not pd.isna(avg_2nd_innings) else 0.0
    
    # Highest and Lowest scores
    # Filter out very short innings or match anomalies if any, but standard is fine
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
        # Find which team batted in this inning
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
            
    # Toss impact at venue
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
        "highest_score": highest_score,
        "highest_team": highest_team,
        "lowest_score": lowest_score,
        "lowest_team": lowest_team,
        "toss_win_match_win_pct": toss_win_match_win_pct
    }

def plot_venue_chasing_vs_defending(stats: Dict[str, Any]) -> go.Figure:
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
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textinfo="value+percent+label")
    return fig

def plot_venue_average_scores(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, limit: int = 10) -> go.Figure:
    """
    Plots the average 1st and 2nd innings scores for the top venues.
    """
    top_venues = matches_df["venue"].value_counts().head(limit).index.tolist()
    
    avg_data = []
    for venue in top_venues:
        match_ids = matches_df[matches_df["venue"] == venue]["id"].tolist()
        v_dels = deliveries_df[deliveries_df["match_id"].isin(match_ids)]
        
        inning_runs = v_dels.groupby(["match_id", "inning"])["total_runs"].sum().reset_index()
        
        avg_1st = inning_runs[inning_runs["inning"] == 1]["total_runs"].mean()
        avg_2nd = inning_runs[inning_runs["inning"] == 2]["total_runs"].mean()
        
        avg_data.append({
            "Venue": venue,
            "1st Innings Average": round(avg_1st, 1) if not pd.isna(avg_1st) else 0,
            "2nd Innings Average": round(avg_2nd, 1) if not pd.isna(avg_2nd) else 0
        })
        
    df = pd.DataFrame(avg_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Venue"],
        y=df["1st Innings Average"],
        name="1st Innings Avg Score",
        marker_color="#e74c3c"
    ))
    fig.add_trace(go.Bar(
        x=df["Venue"],
        y=df["2nd Innings Average"],
        name="2nd Innings Avg Score",
        marker_color="#3498db"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=f"Average Innings Scores at Top {limit} Venues",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Average Score"),
        barmode="group",
        height=450,
        margin=dict(l=20, r=20, t=50, b=80)
    )
    return fig
