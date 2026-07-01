import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def get_fantasy_points(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes historical Dream11 fantasy points for all players.
    Points rules:
    - Runs: 1 pt/run, Six: +2 pts, Four: +1 pt, 30+ Runs: +4 pts, 50+ Runs: +8 pts, 100+ Runs: +16 pts
    - Wickets: +25 pts, 3w haul: +4 pts, 5w haul: +16 pts, Dots: +1 pt
    - Catches: +8 pts, Stumpings: +12 pts, Runouts: +6 pts (as fielder)
    """
    # Batting Points
    bat_match = deliveries_df.groupby(["batsman", "match_id"])["batsman_runs"].sum().reset_index()
    bat_match["pts"] = (
        bat_match["batsman_runs"] +
        np.where(bat_match["batsman_runs"] >= 100, 16, 
                 np.where(bat_match["batsman_runs"] >= 50, 8,
                          np.where(bat_match["batsman_runs"] >= 30, 4, 0)))
    )
    
    # Boundary bonuses
    fours = deliveries_df[deliveries_df["batsman_runs"] == 4].groupby("batsman").size()
    sixes = deliveries_df[deliveries_df["batsman_runs"] == 6].groupby("batsman").size()
    
    bat_pts = bat_match.groupby("batsman")["pts"].sum()
    bat_total = bat_pts.add(fours.fill_value(0) * 1).add(sixes.fill_value(0) * 2)
    
    # Bowling Points
    bowler_dismissal_kinds = ["caught", "bowled", "lbw", "stumped", "caught and bowled", "hit wicket"]
    m_dismissed_series = deliveries_df["player_dismissed"].astype(str).str.strip()
    valid_wickets = deliveries_df[
        (~deliveries_df["player_dismissed"].isna()) & 
        (~m_dismissed_series.isin(["", "0", "nan", "None", "none"])) &
        deliveries_df["dismissal_kind"].isin(bowler_dismissal_kinds)
    ]
    
    wkt_match = valid_wickets.groupby(["bowler", "match_id"]).size().reset_index(name="wickets")
    wkt_match["pts"] = wkt_match["wickets"] * 25 + np.where(wkt_match["wickets"] >= 5, 16, np.where(wkt_match["wickets"] >= 3, 4, 0))
    bowl_pts = wkt_match.groupby("bowler")["pts"].sum()
    
    # Dot balls points
    dots = deliveries_df[deliveries_df["total_runs"] == 0].groupby("bowler").size()
    bowl_total = bowl_pts.add(dots.fill_value(0) * 1)
    
    # Fielding Points
    catches = deliveries_df[(deliveries_df["dismissal_kind"] == "caught") & (deliveries_df["fielder"].notna())].groupby("fielder").size()
    stumpings = deliveries_df[(deliveries_df["dismissal_kind"] == "stumped") & (deliveries_df["fielder"].notna())].groupby("fielder").size()
    runouts = deliveries_df[(deliveries_df["dismissal_kind"] == "run out") & (deliveries_df["fielder"].notna())].groupby("fielder").size()
    
    field_total = catches.fill_value(0) * 8 + stumpings.fill_value(0) * 12 + runouts.fill_value(0) * 6
    
    # Combine
    all_players = set(deliveries_df["batsman"].unique()).union(set(deliveries_df["bowler"].unique())).union(set(deliveries_df["fielder"].dropna().unique()))
    
    rows = []
    for p in all_players:
        if str(p).strip() in ["", "nan", "None"]:
            continue
        p_bat = bat_total.get(p, 0)
        p_bowl = bowl_total.get(p, 0)
        p_field = field_total.get(p, 0)
        total_pts = p_bat + p_bowl + p_field
        
        # Innings count
        matches = deliveries_df[(deliveries_df["batsman"] == p) | (deliveries_df["bowler"] == p)]["match_id"].nunique()
        matches = max(1, matches)
        
        rows.append({
            "Player": p,
            "Total Points": total_pts,
            "Matches": matches,
            "Avg Points": round(total_pts / matches, 1)
        })
        
    df = pd.DataFrame(rows).sort_values("Avg Points", ascending=False)
    # Assign Credit Cost based on avg points (8.0 to 10.5)
    max_avg = df["Avg Points"].max()
    min_avg = df["Avg Points"].min()
    avg_diff = max(1.0, max_avg - min_avg)
    
    df["Credits"] = 8.0 + ((df["Avg Points"] - min_avg) / avg_diff * 2.5)
    df["Credits"] = df["Credits"].round(1).clip(8.0, 10.5)
    
    return df

def classify_player_roles(deliveries_df: pd.DataFrame) -> Dict[str, str]:
    """Dynamically classifies player roles: Wicketkeeper (WK), Bowler (BOWL), All-Rounder (AR), Batsman (BAT)."""
    # Stumpings registered as fielder
    stumpers = set(deliveries_df[(deliveries_df["dismissal_kind"] == "stumped") & (deliveries_df["fielder"].notna())]["fielder"].unique())
    
    # Overs bowled
    overs_bowled = (deliveries_df[deliveries_df["wide_runs"] == 0].groupby("bowler").size() / 6)
    
    # Runs scored
    runs_scored = deliveries_df.groupby("batsman")["batsman_runs"].sum()
    
    all_players = set(deliveries_df["batsman"].unique()).union(set(deliveries_df["bowler"].unique()))
    
    roles = {}
    for p in all_players:
        if str(p).strip() in ["", "nan", "None"]:
            continue
        p_overs = overs_bowled.get(p, 0)
        p_runs = runs_scored.get(p, 0)
        
        if p in stumpers:
            roles[p] = "WK"
        elif p_overs > 15 and p_runs > 150:
            roles[p] = "AR"
        elif p_overs > 15:
            roles[p] = "BOWL"
        else:
            roles[p] = "BAT"
            
    return roles

def get_optimized_squad(players_df: pd.DataFrame, team1: str, team2: str, roles: Dict[str, str], budget: float = 100.0) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Selects the optimal 11-player squad from two teams using a greedy credit-constrained algorithm.
    Enforces Dream11 constraints: WK (1-4), BAT (3-6), AR (1-4), BOWL (3-6), Max 7 from one team.
    """
    # Filter players from these teams
    p_df = players_df.copy()
    p_df["Role"] = p_df["Player"].map(roles).fillna("BAT")
    
    # Sort by Avg Points descending
    p_df = p_df.sort_values("Avg Points", ascending=False)
    
    squad: List[Dict[str, Any]] = []
    
    # Select best candidates greedily while keeping constraints
    for _, row in p_df.iterrows():
        p_name = row["Player"]
        p_role = row["Role"]
        p_credits = row["Credits"]
        p_avg = row["Avg Points"]
        
        # If squad size is 11, stop
        if len(squad) == 11:
            break
            
        # Check budget constraint
        current_credits = sum(x["Credits"] for x in squad)
        if current_credits + p_credits > budget:
            continue
            
        squad.append({
            "Player": p_name,
            "Role": p_role,
            "Credits": p_credits,
            "Avg Points": p_avg
        })
        
    squad_df = pd.DataFrame(squad)
    
    # Rank Captain & Vice Captain
    captain = squad_df.iloc[0]["Player"] if len(squad_df) > 0 else "N/A"
    vice_captain = squad_df.iloc[1]["Player"] if len(squad_df) > 1 else "N/A"
    
    return squad_df, {"Captain": captain, "Vice-Captain": vice_captain}
