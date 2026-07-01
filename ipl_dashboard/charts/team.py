import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from constants.teams import get_team_color

def plot_team_wins(team_stats: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots a horizontal bar chart of overall match wins by team."""
    fig = px.bar(
        team_stats, x="Matches Won", y="Team", orientation="h",
        title="Overall Matches Won by Teams",
        color="Team", color_discrete_map={t: get_team_color(t) for t in team_stats["Team"]},
        text="Matches Won", hover_data=["Matches Played", "Win Percentage (%)"]
    )
    fig.update_layout(
        template=template, yaxis={"categoryorder": "total ascending"},
        xaxis_title="Wins", yaxis_title=None, showlegend=False,
        height=450, margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="inside", marker_line_color="rgba(0,0,0,0.2)", marker_line_width=1)
    return fig

def plot_season_performance(performance: pd.DataFrame, team: str, template: str = "plotly_dark") -> go.Figure:
    """Plots win rates and matches played by season for a team."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=performance["season"], y=performance["Played"], name="Played",
        marker_color="rgba(100, 149, 237, 0.6)", hovertemplate="Season: %{x}<br>Played: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=performance["season"], y=performance["Wins"], name="Won",
        marker_color=get_team_color(team), hovertemplate="Season: %{x}<br>Won: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=performance["season"], y=performance["Win Rate (%)"], name="Win Rate %",
        yaxis="y2", mode="lines+markers", line=dict(color="orange", width=3),
        marker=dict(size=8), hovertemplate="Season: %{x}<br>Win Rate: %{y}%<extra></extra>"
    ))
    fig.update_layout(
        template=template, title=f"Season Performance - {team}",
        xaxis=dict(title="Season", type="category"), yaxis=dict(title="Matches"),
        yaxis2=dict(title="Win Rate (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"), barmode="group",
        height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_toss_impact(labels: List[str], values: List[float], team: str, template: str = "plotly_dark") -> go.Figure:
    """Plots the toss impact pie chart."""
    fig = px.pie(
        names=labels, values=values, title=f"Toss Impact on Match Outcomes ({team})",
        color_discrete_sequence=["#1abc9c", "#e74c3c"], hole=0.4
    )
    fig.update_layout(template=template, height=350, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_traces(textinfo="percent+label")
    return fig

def plot_toss_decisions(toss_dec_df: pd.DataFrame, team: str, template: str = "plotly_dark") -> go.Figure:
    """Plots toss decisions (Bat vs Field)."""
    fig = px.pie(
        toss_dec_df, names="Decision", values="Count", title=f"Toss Decisions ({team})",
        color="Decision", color_discrete_map={"field": "#3498db", "bat": "#f1c40f"}, hole=0.4
    )
    fig.update_layout(template=template, height=350, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_traces(textinfo="value+percent+label")
    return fig

def plot_team_strength_radar(categories: List[str], team_vals: List[float], avg_vals: List[float], team_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots a polar radar chart comparing a team's attributes against league averages."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=team_vals, theta=categories, fill='toself',
        name=team_name, line_color=get_team_color(team_name)
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_vals, theta=categories, fill='toself',
        name="League Average", line_color="#7f8c8d"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template=template, title=f"Team Strength Radar Comparison - {team_name}",
        showlegend=True, height=400, margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
