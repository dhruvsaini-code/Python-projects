import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def get_team_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes overall statistics for all teams: Matches played, won, lost, and win rate.
    """
    # Count matches as team1 or team2
    t1_counts = matches_df["team1"].value_counts()
    t2_counts = matches_df["team2"].value_counts()
    
    # Combine counts for total matches
    total_played = t1_counts.add(t2_counts, fill_value=0).astype(int)
    
    # Matches won
    wins = matches_df[matches_df["winner"] != "No Result"]["winner"].value_counts().astype(int)
    
    # Combine into stats dataframe
    stats = pd.DataFrame({
        "Matches Played": total_played,
        "Matches Won": wins
    }).fillna(0)
    
    stats["Matches Played"] = stats["Matches Played"].astype(int)
    stats["Matches Won"] = stats["Matches Won"].astype(int)
    stats["Matches Lost"] = stats["Matches Played"] - stats["Matches Won"]
    stats["Win Percentage (%)"] = ((stats["Matches Won"] / stats["Matches Played"]) * 100).round(2)
    
    stats = stats.reset_index().rename(columns={"index": "Team"})
    return stats.sort_values(by="Win Percentage (%)", ascending=False)

def get_head_to_head_stats(matches_df: pd.DataFrame, team1: str, team2: str) -> Dict[str, Any]:
    """
    Computes head-to-head statistics between two teams.
    """
    h2h_matches = matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ]
    
    total = len(h2h_matches)
    if total == 0:
        return {
            "total": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "no_result": 0,
            "team1_win_pct": 0.0,
            "team2_win_pct": 0.0
        }
        
    t1_wins = len(h2h_matches[h2h_matches["winner"] == team1])
    t2_wins = len(h2h_matches[h2h_matches["winner"] == team2])
    no_result = total - (t1_wins + t2_wins)
    
    return {
        "total": total,
        "team1_wins": t1_wins,
        "team2_wins": t2_wins,
        "no_result": no_result,
        "team1_win_pct": round((t1_wins / total) * 100, 2),
        "team2_win_pct": round((t2_wins / total) * 100, 2)
    }

def plot_team_wins(matches_df: pd.DataFrame) -> go.Figure:
    """
    Plots a beautiful bar chart of total match wins by team.
    """
    stats_df = get_team_stats(matches_df)
    
    fig = px.bar(
        stats_df, 
        x="Matches Won", 
        y="Team",
        orientation="h",
        title="Overall Matches Won by Teams",
        color="Matches Won",
        color_continuous_scale="Viridis",
        text="Matches Won",
        hover_data=["Matches Played", "Win Percentage (%)"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Wins",
        yaxis_title=None,
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_traces(
        textposition="inside",
        texttemplate="%{text}",
        marker_line_color="rgb(8,48,107)",
        marker_line_width=1.5,
        opacity=0.9
    )
    return fig

def plot_season_performance(matches_df: pd.DataFrame, team: str) -> go.Figure:
    """
    Plots win progression over seasons for a specific team.
    """
    # Filter for matches where the team played
    team_matches = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)]
    
    # Calculate matches played per season
    played_per_season = team_matches.groupby("season").size().reset_index(name="Played")
    
    # Calculate wins per season
    wins_per_season = team_matches[team_matches["winner"] == team].groupby("season").size().reset_index(name="Wins")
    
    # Merge and calculate win percentage
    performance = pd.merge(played_per_season, wins_per_season, on="season", how="outer").fillna(0)
    performance["Wins"] = performance["Wins"].astype(int)
    performance["Win Rate (%)"] = ((performance["Wins"] / performance["Played"]) * 100).round(1)
    
    fig = go.Figure()
    
    # Add matches played bar
    fig.add_trace(go.Bar(
        x=performance["season"],
        y=performance["Played"],
        name="Matches Played",
        marker_color="rgba(100, 149, 237, 0.6)",
        hoverinfo="y+name"
    ))
    
    # Add matches won bar
    fig.add_trace(go.Bar(
        x=performance["season"],
        y=performance["Wins"],
        name="Matches Won",
        marker_color="rgba(46, 204, 113, 0.8)",
        hoverinfo="y+name"
    ))
    
    # Add Win rate line
    fig.add_trace(go.Scatter(
        x=performance["season"],
        y=performance["Win Rate (%)"],
        name="Win Rate (%)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="orange", width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=f"Season-wise Performance - {team}",
        xaxis=dict(title="Season", type="category"),
        yaxis=dict(title="Number of Matches"),
        yaxis2=dict(
            title="Win Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        barmode="group",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_toss_impact(matches_df: pd.DataFrame, team: str = "All") -> go.Figure:
    """
    Plots the impact of winning the toss on match win percentage.
    """
    df = matches_df.copy()
    if team != "All":
        df = df[(df["team1"] == team) | (df["team2"] == team)]
        
    total_matches = len(df)
    if total_matches == 0:
        return go.Figure()
        
    # How often did the toss winner win the match?
    toss_and_match_win = df[df["toss_winner"] == df["winner"]]
    toss_win_match_win_pct = len(toss_and_match_win) / total_matches
    
    labels = ["Toss Winner Wins Match", "Toss Winner Loses Match"]
    values = [toss_win_match_win_pct, 1 - toss_win_match_win_pct]
    
    fig = px.pie(
        names=labels,
        values=values,
        title=f"Toss Impact on Match Outcomes ({team})",
        color_discrete_sequence=["#1abc9c", "#e74c3c"],
        hole=0.4
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textinfo="percent+label")
    return fig

def plot_toss_decisions(matches_df: pd.DataFrame, team: str = "All") -> go.Figure:
    """
    Plots toss decisions (Bat vs Field).
    """
    df = matches_df.copy()
    if team != "All":
        df = df[df["toss_winner"] == team]
        
    toss_decisions = df["toss_decision"].value_counts().reset_index()
    toss_decisions.columns = ["Decision", "Count"]
    
    fig = px.pie(
        toss_decisions,
        names="Decision",
        values="Count",
        title=f"Toss Decisions ({team})",
        color="Decision",
        color_discrete_map={"field": "#3498db", "bat": "#f1c40f"},
        hole=0.4
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textinfo="value+percent+label")
    return fig
