import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

def plot_venue_chasing_vs_defending(stats: Dict[str, Any], template: str = "plotly_dark") -> go.Figure:
    """Plots a pie chart of chasing vs defending wins for a stadium."""
    labels = ["Chasing Team Wins", "Defending Team Wins", "No Result"]
    values = [stats["chasing_wins"], stats["defending_wins"], stats["no_results"]]
    
    fig = px.pie(
        names=labels, values=values, title=f"Match Outcome Distribution - {stats['venue']}",
        color_discrete_sequence=["#2980b9", "#27ae60", "#7f8c8d"], hole=0.4
    )
    fig.update_layout(template=template, height=350, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_traces(textinfo="value+percent+label")
    return fig

def plot_venue_average_scores(comp_df: pd.DataFrame, limit: int = 10, template: str = "plotly_dark") -> go.Figure:
    """Plots comparative average 1st and 2nd innings scores for the top venues."""
    df = comp_df.head(limit)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Venue"], y=df["Avg 1st Innings"], name="1st Innings", marker_color="#e74c3c"
    ))
    fig.add_trace(go.Bar(
        x=df["Venue"], y=df["Avg 2nd Innings"], name="2nd Innings", marker_color="#3498db"
    ))
    fig.update_layout(
        template=template, title=f"Average Innings Scores at Top {limit} Venues",
        xaxis=dict(tickangle=-45), yaxis=dict(title="Average Score"),
        barmode="group", height=400, margin=dict(l=20, r=20, t=50, b=80)
    )
    return fig

def plot_venue_phase_scores(phases: List[str], scores: List[float], venue_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots a bar chart comparing powerplay, middle, and death overs average runs at a venue."""
    fig = px.bar(
        x=phases, y=scores, title=f"Average Phase Scores - {venue_name}",
        color=scores, color_continuous_scale="Teal", text=scores
    )
    fig.update_layout(
        template=template, xaxis_title=None, yaxis_title="Average Runs",
        coloraxis_showscale=False, height=350, margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="outside")
    return fig
