import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List

def plot_top_batsmen(top_df: pd.DataFrame, limit: int = 10, template: str = "plotly_dark") -> go.Figure:
    """Plots top batsmen by runs."""
    fig = px.bar(
        top_df, x="Runs", y="Player", orientation="h",
        title=f"Top {limit} Batsmen - Most Runs in IPL",
        color="Runs", color_continuous_scale="Oranges",
        text="Runs", hover_data=["Innings", "Strike Rate", "Average"]
    )
    fig.update_layout(
        template=template, yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False, height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="inside")
    return fig

def plot_top_bowlers(top_df: pd.DataFrame, limit: int = 10, template: str = "plotly_dark") -> go.Figure:
    """Plots top bowlers by wickets."""
    fig = px.bar(
        top_df, x="Wickets", y="Player", orientation="h",
        title=f"Top {limit} Bowlers - Most Wickets in IPL",
        color="Wickets", color_continuous_scale="Purples",
        text="Wickets", hover_data=["Innings", "Overs", "Economy", "Average"]
    )
    fig.update_layout(
        template=template, yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False, height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="inside")
    return fig

def plot_batsman_scatter(filtered: pd.DataFrame, min_runs: int = 500, template: str = "plotly_dark") -> go.Figure:
    """Scatter plot: Strike Rate vs Average."""
    fig = px.scatter(
        filtered, x="Average", y="Strike Rate", size="Runs", color="Runs",
        hover_name="Player", hover_data=["Innings", "Highest Score", "50s", "100s"],
        title=f"Batting Strike Rate vs Average (Min {min_runs} runs)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template=template, height=450, margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Batting Average", yaxis_title="Strike Rate"
    )
    return fig

def plot_bowler_scatter(filtered: pd.DataFrame, min_wickets: int = 30, template: str = "plotly_dark") -> go.Figure:
    """Scatter plot: Economy vs Wickets."""
    fig = px.scatter(
        filtered, x="Wickets", y="Economy", size="Overs", color="Economy",
        hover_name="Player", hover_data=["Innings", "Average", "Strike Rate", "Dot Ball Pct (%)"],
        title=f"Bowling Economy vs Wickets (Min {min_wickets} wickets)",
        color_continuous_scale="Plasma_r"
    )
    fig.update_layout(
        template=template, height=450, margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Total Wickets", yaxis_title="Economy Rate (Lower is better)"
    )
    return fig

def plot_strike_rate_evolution(seasons: List[str], strike_rates: List[float], player_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots a line chart of strike rate evolution over seasons."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seasons, y=strike_rates, mode="lines+markers",
        line=dict(color="#f97316", width=3), marker=dict(size=8, color="#ea580c"),
        hovertemplate="Season: %{x}<br>Strike Rate: %{y}<extra></extra>"
    ))
    fig.update_layout(
        template=template, title=f"Strike Rate Evolution - {player_name}",
        xaxis_title="Season", yaxis_title="Strike Rate", height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_wicket_distribution(kinds: List[str], counts: List[int], player_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots a pie chart of dismissal kinds for a bowler."""
    fig = px.pie(
        names=kinds, values=counts, title=f"Wicket Distribution - {player_name}",
        color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3
    )
    fig.update_layout(template=template, height=350, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_traces(textinfo="percent+label")
    return fig

def plot_career_runs_progression(matches: List[int], cumulative_runs: List[int], player_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots career runs progression matching timeline."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=matches, y=cumulative_runs, mode="lines", fill="tozeroy",
        line=dict(color="#3b82f6", width=2), hovertemplate="Match Count: %{x}<br>Cumulative Runs: %{y}<extra></extra>"
    ))
    fig.update_layout(
        template=template, title=f"Career Runs Progression - {player_name}",
        xaxis_title="Matches Played", yaxis_title="Cumulative Runs", height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_player_comparison_radar(categories: List[str], vals_a: List[float], vals_b: List[float], name_a: str, name_b: str, template: str = "plotly_dark") -> go.Figure:
    """Radar chart comparing two players across normalized dimensions."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill='toself', name=name_a, line_color="#3498db"))
    fig.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill='toself', name=name_b, line_color="#e74c3c"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template=template, title=f"Player Head-to-Head - {name_a} vs {name_b}",
        showlegend=True, height=400, margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
