import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List

def get_batsman_stats(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes batting statistics for all players.
    """
    # 1. Total Runs
    runs = deliveries_df.groupby("batsman")["batsman_runs"].sum()
    
    # 2. Balls Faced (excluding wides)
    balls = deliveries_df[deliveries_df["wide_runs"] == 0].groupby("batsman").size()
    
    # 3. Innings Played
    innings = deliveries_df.groupby("batsman")["match_id"].nunique()
    
    # 4. Dismissals
    dismissals = deliveries_df[deliveries_df["player_dismissed"].notna()].groupby("player_dismissed").size()
    
    # 5. Boundaries (4s and 6s)
    fours = deliveries_df[deliveries_df["batsman_runs"] == 4].groupby("batsman").size()
    sixes = deliveries_df[deliveries_df["batsman_runs"] == 6].groupby("batsman").size()
    
    # Compile into a DataFrame
    df = pd.DataFrame({
        "Innings": innings,
        "Runs": runs,
        "Balls Faced": balls,
        "Dismissals": dismissals,
        "Fours": fours,
        "Sixes": sixes
    }).fillna(0)
    
    # Cast types
    df = df.astype(int)
    
    # Compute derived metrics
    df["Strike Rate"] = np.where(df["Balls Faced"] > 0, (df["Runs"] / df["Balls Faced"]) * 100, 0.0).round(2)
    # Average = Runs / Dismissals (or Runs if dismissals == 0)
    df["Average"] = np.where(df["Dismissals"] > 0, df["Runs"] / df["Dismissals"], df["Runs"]).astype(float).round(2)
    
    # Boundary Percentage
    df["Boundary Runs"] = df["Fours"] * 4 + df["Sixes"] * 6
    df["Boundary Pct (%)"] = np.where(df["Runs"] > 0, (df["Boundary Runs"] / df["Runs"]) * 100, 0.0).round(2)
    
    df = df.reset_index().rename(columns={"batsman": "Player"})
    return df

def get_bowler_stats(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes bowling statistics for all players.
    """
    # Exclude run outs, retired hurts, obstructing the field for bowler wickets
    bowler_dismissal_kinds = [
        "caught", "bowled", "lbw", "stumped", "caught and bowled", "hit wicket"
    ]
    
    # 1. Wickets
    wickets = deliveries_df[
        deliveries_df["player_dismissed"].notna() & 
        deliveries_df["dismissal_kind"].isin(bowler_dismissal_kinds)
    ].groupby("bowler").size()
    
    # 2. Balls Bowled (exclude wides)
    balls = deliveries_df[deliveries_df["wide_runs"] == 0].groupby("bowler").size()
    
    # 3. Runs conceded (batsman runs + wide + noball)
    deliveries_df_copy = deliveries_df.copy()
    deliveries_df_copy["runs_conceded"] = (
        deliveries_df_copy["batsman_runs"] + 
        deliveries_df_copy["wide_runs"] + 
        deliveries_df_copy["noball_runs"]
    )
    runs_conceded = deliveries_df_copy.groupby("bowler")["runs_conceded"].sum()
    
    # 4. Innings played
    innings = deliveries_df.groupby("bowler")["match_id"].nunique()
    
    # Compile
    df = pd.DataFrame({
        "Innings": innings,
        "Wickets": wickets,
        "Balls Bowled": balls,
        "Runs Conceded": runs_conceded
    }).fillna(0)
    
    df = df.astype(int)
    
    # Derived metrics
    df["Overs"] = (df["Balls Bowled"] / 6).round(1)
    df["Economy"] = np.where(df["Balls Bowled"] > 0, (df["Runs Conceded"] / (df["Balls Bowled"] / 6)), 0.0).round(2)
    df["Average"] = np.where(df["Wickets"] > 0, df["Runs Conceded"] / df["Wickets"], 0.0).round(2)
    df["Strike Rate"] = np.where(df["Wickets"] > 0, df["Balls Bowled"] / df["Wickets"], 0.0).round(2)
    
    df = df.reset_index().rename(columns={"bowler": "Player"})
    return df

def get_orange_cap_trends(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the Orange Cap (most runs) winner for each season.
    """
    merged = deliveries_df.merge(matches_df[["id", "season"]], left_on="match_id", right_on="id")
    season_runs = merged.groupby(["season", "batsman"])["batsman_runs"].sum().reset_index()
    
    # Sort and get highest for each season
    orange_caps = season_runs.sort_values("batsman_runs", ascending=False).drop_duplicates("season")
    orange_caps = orange_caps.sort_values("season").rename(columns={"batsman": "Player", "batsman_runs": "Runs"})
    return orange_caps

def get_purple_cap_trends(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the Purple Cap (most wickets) winner for each season.
    """
    bowler_dismissal_kinds = ["caught", "bowled", "lbw", "stumped", "caught and bowled", "hit wicket"]
    merged = deliveries_df[
        deliveries_df["player_dismissed"].notna() & 
        deliveries_df["dismissal_kind"].isin(bowler_dismissal_kinds)
    ].merge(matches_df[["id", "season"]], left_on="match_id", right_on="id")
    
    season_wickets = merged.groupby(["season", "bowler"]).size().reset_index(name="Wickets")
    
    # Sort and get highest for each season
    purple_caps = season_wickets.sort_values("Wickets", ascending=False).drop_duplicates("season")
    purple_caps = purple_caps.sort_values("season").rename(columns={"bowler": "Player"})
    return purple_caps

def get_player_profile(player_name: str, deliveries_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a complete statistics profile for a specific player (batting and bowling).
    """
    # 1. Batting Details
    batting_del = deliveries_df[deliveries_df["batsman"] == player_name]
    has_batted = len(batting_del) > 0
    
    batting_profile = {}
    if has_batted:
        runs = batting_del["batsman_runs"].sum()
        balls = batting_del[batting_del["wide_runs"] == 0].shape[0]
        innings = batting_del["match_id"].nunique()
        dismissals = deliveries_df[deliveries_df["player_dismissed"] == player_name].shape[0]
        fours = batting_del[batting_del["batsman_runs"] == 4].shape[0]
        sixes = batting_del[batting_del["batsman_runs"] == 6].shape[0]
        
        # Calculate scores per match to find 50s and 100s
        match_runs = batting_del.groupby("match_id")["batsman_runs"].sum()
        fifties = ((match_runs >= 50) & (match_runs < 100)).sum()
        hundreds = (match_runs >= 100).sum()
        highest_score = match_runs.max()
        
        strike_rate = round((runs / balls * 100), 2) if balls > 0 else 0.0
        average = round((runs / dismissals), 2) if dismissals > 0 else runs
        
        batting_profile = {
            "Innings": innings,
            "Runs": runs,
            "Balls Faced": balls,
            "Average": average,
            "Strike Rate": strike_rate,
            "Highest Score": highest_score,
            "Fours": fours,
            "Sixes": sixes,
            "50s": fifties,
            "100s": hundreds
        }
        
    # 2. Bowling Details
    bowling_del = deliveries_df[deliveries_df["bowler"] == player_name]
    has_bowled = len(bowling_del) > 0
    
    bowling_profile = {}
    if has_bowled:
        bowler_dismissal_kinds = ["caught", "bowled", "lbw", "stumped", "caught and bowled", "hit wicket"]
        wickets = bowling_del[
            bowling_del["player_dismissed"].notna() & 
            bowling_del["dismissal_kind"].isin(bowler_dismissal_kinds)
        ].shape[0]
        
        balls_bowled = bowling_del[bowling_del["wide_runs"] == 0].shape[0]
        innings_bowled = bowling_del["match_id"].nunique()
        
        runs_conceded = (
            bowling_del["batsman_runs"].sum() + 
            bowling_del["wide_runs"].sum() + 
            bowling_del["noball_runs"].sum()
        )
        
        # Best Bowling Figures
        bowling_del_copy = bowling_del.copy()
        bowling_del_copy["runs_conceded_single"] = (
            bowling_del_copy["batsman_runs"] + 
            bowling_del_copy["wide_runs"] + 
            bowling_del_copy["noball_runs"]
        )
        
        match_wickets = bowling_del_copy[
            bowling_del_copy["player_dismissed"].notna() & 
            bowling_del_copy["dismissal_kind"].isin(bowler_dismissal_kinds)
        ].groupby("match_id").size()
        
        match_runs = bowling_del_copy.groupby("match_id")["runs_conceded_single"].sum()
        
        # Merge to find best figures
        match_bowling = pd.DataFrame({"Wickets": match_wickets, "Runs": match_runs}).fillna({"Wickets": 0}).astype(int)
        
        best_figures_str = "N/A"
        if not match_bowling.empty:
            # Sort by Wickets (desc) and Runs (asc)
            best_match = match_bowling.sort_values(by=["Wickets", "Runs"], ascending=[False, True]).iloc[0]
            best_figures_str = f"{best_match['Wickets']}/{best_match['Runs']}"
            
        # Count 3w and 5w hauls
        three_w = (match_bowling["Wickets"] >= 3).sum()
        five_w = (match_bowling["Wickets"] >= 5).sum()
        
        overs = round(balls_bowled / 6, 1)
        economy = round(runs_conceded / (balls_bowled / 6), 2) if balls_bowled > 0 else 0.0
        average = round(runs_conceded / wickets, 2) if wickets > 0 else 0.0
        strike_rate = round(balls_bowled / wickets, 2) if wickets > 0 else 0.0
        
        bowling_profile = {
            "Innings": innings_bowled,
            "Wickets": wickets,
            "Overs": overs,
            "Economy": economy,
            "Average": average,
            "Strike Rate": strike_rate,
            "Best Figures": best_figures_str,
            "3W Hauls": three_w,
            "5W Hauls": five_w
        }
        
    return {
        "name": player_name,
        "has_batted": has_batted,
        "has_bowled": has_bowled,
        "batting": batting_profile,
        "bowling": bowling_profile
    }

def plot_top_batsmen(batsman_df: pd.DataFrame, limit: int = 10) -> go.Figure:
    """
    Plots top batsmen by runs.
    """
    top_df = batsman_df.sort_values("Runs", ascending=False).head(limit)
    
    fig = px.bar(
        top_df,
        x="Runs",
        y="Player",
        orientation="h",
        title=f"Top {limit} Batsmen - Most Runs in IPL",
        color="Runs",
        color_continuous_scale="Oranges",
        text="Runs",
        hover_data=["Innings", "Strike Rate", "Average"]
    )
    fig.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="inside")
    return fig

def plot_top_bowlers(bowler_df: pd.DataFrame, limit: int = 10) -> go.Figure:
    """
    Plots top bowlers by wickets.
    """
    top_df = bowler_df.sort_values("Wickets", ascending=False).head(limit)
    
    fig = px.bar(
        top_df,
        x="Wickets",
        y="Player",
        orientation="h",
        title=f"Top {limit} Bowlers - Most Wickets in IPL",
        color="Wickets",
        color_continuous_scale="Purples",
        text="Wickets",
        hover_data=["Innings", "Overs", "Economy", "Average"]
    )
    fig.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="inside")
    return fig
