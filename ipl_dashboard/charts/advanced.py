import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List

def plot_correlation_matrix(corr_df: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots a correlation heatmap for numerical match indicators."""
    fig = px.imshow(
        corr_df, text_auto=".2f", aspect="auto",
        labels=dict(color="Correlation"),
        x=[c.replace('_', ' ').title() for c in corr_df.columns],
        y=[c.replace('_', ' ').title() for c in corr_df.index],
        color_continuous_scale="RdBu_r", title="Match Outcome Metrics Correlation Matrix"
    )
    fig.update_layout(template=template, height=400, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_pca_clustering(pca_df: pd.DataFrame, player_type: str = "batsman", template: str = "plotly_dark") -> go.Figure:
    """Plots a 2D scatter plot showing K-Means clusters of players based on PCA components."""
    size_col = "Runs" if player_type == "batsman" else "Wickets"
    fig = px.scatter(
        pca_df, x="PCA1", y="PCA2", color="Cluster Role", size=size_col,
        hover_name="Player", hover_data=["Innings", "Average"],
        title=f"K-Means Clustering & PCA Mapping of IPL {player_type.title()}s",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(template=template, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_team_similarity(similarity_df: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots a similarity heatmap between IPL teams based on pairwise cosine similarity."""
    fig = px.imshow(
        similarity_df, text_auto=".2f", aspect="auto",
        color_continuous_scale="Viridis", title="Team Profile Cosine Similarity Heatmap"
    )
    fig.update_layout(template=template, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_player_bubble_chart(filtered_df: pd.DataFrame, min_runs: int = 500, template: str = "plotly_dark") -> go.Figure:
    """Plots player Strike Rate vs Average bubble chart with color as Boundary %."""
    fig = px.scatter(
        filtered_df, x="Average", y="Strike Rate", size="Runs", color="Boundary Pct (%)",
        hover_name="Player", hover_data=["Innings", "Fours", "Sixes"],
        title=f"Batsman Impact Bubble Chart (Min {min_runs} runs)",
        color_continuous_scale="Reds"
    )
    fig.update_layout(template=template, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_sankey_flow(source_labels: List[str], target_labels: List[str], values: List[int], node_labels: List[str], template: str = "plotly_dark") -> go.Figure:
    """Plots a Sankey flow diagram showing Toss Winner -> Toss Decision -> Winner mapping."""
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=node_labels, color="rgba(52, 152, 219, 0.8)"),
        link=dict(source=source_labels, target=target_labels, value=values, color="rgba(255, 255, 255, 0.15)")
    )])
    fig.update_layout(template=template, title_text="Toss Action to Winner Sankey Flow Map", font_size=10, height=400, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_sunburst_outcomes(matches_df: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots a Sunburst chart of Winner -> Season -> Venue distribution hierarchy."""
    df = matches_df[matches_df["winner"] != "No Result"].copy()
    fig = px.sunburst(
        df, path=["winner", "season", "city"],
        title="IPL Match Outcomes Distribution Hierarchy (Sunburst)",
        color_continuous_scale="RdBu"
    )
    fig.update_layout(template=template, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_treemap_runs(deliveries_df: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots a Treemap of runs contributed by batting teams and batsmen."""
    # Find top 8 teams and top 5 batsmen per team
    team_runs = deliveries_df.groupby("batting_team")["batsman_runs"].sum().reset_index()
    top_teams = team_runs.sort_values("batsman_runs", ascending=False).head(8)["batting_team"].tolist()
    
    df_filtered = deliveries_df[deliveries_df["batting_team"].isin(top_teams)]
    p_runs = df_filtered.groupby(["batting_team", "batsman"])["batsman_runs"].sum().reset_index()
    
    top_p = p_runs.sort_values(["batting_team", "batsman_runs"], ascending=[True, False]).groupby("batting_team").head(5)
    
    fig = px.treemap(
        top_p, path=["batting_team", "batsman"], values="batsman_runs",
        title="Runs Distribution Map (Top Teams & Top 5 Batsmen)",
        color="batsman_runs", color_continuous_scale="Blues"
    )
    fig.update_layout(template=template, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_parallel_coordinates(df: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots a Parallel Coordinates chart comparing multi-dimensional player stats."""
    cols = ["Innings", "Runs", "Average", "Strike Rate", "Boundary Pct (%)", "Consistency Score"]
    fig = px.parallel_coordinates(
        df, columns=cols, color="Runs",
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Batsmen Career Metric Axes (Parallel Coordinates)"
    )
    fig.update_layout(template=template, height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_season_timeline_animation(season_comp_df: pd.DataFrame, template: str = "plotly_dark") -> go.Figure:
    """Plots an animated line timeline of season runs/boundary updates."""
    fig = px.line(
        season_comp_df, x="Season", y="Avg Runs/Match",
        title="Historical Season Average Score Timeline", markers=True
    )
    fig.update_layout(template=template, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig
