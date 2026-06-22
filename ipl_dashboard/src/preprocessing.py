import pandas as pd
import numpy as np
from typing import Tuple, List

def get_unique_seasons(matches_df: pd.DataFrame) -> List[int]:
    """
    Returns sorted list of unique seasons from matches.
    """
    if "season" not in matches_df.columns:
        return []
    return sorted(matches_df["season"].dropna().unique().astype(int).tolist(), reverse=True)

def get_unique_teams(matches_df: pd.DataFrame) -> List[str]:
    """
    Returns sorted list of unique teams from matches.
    """
    # Exclude No Result or other filler names
    teams1 = set(matches_df["team1"].dropna().unique())
    teams2 = set(matches_df["team2"].dropna().unique())
    teams = teams1.union(teams2)
    # Filter out anything that is not a proper team name (e.g. empty or No Result)
    teams = {t for t in teams if str(t).strip() not in ["", "No Result", "Unknown"]}
    return sorted(list(teams))

def get_unique_venues(matches_df: pd.DataFrame) -> List[str]:
    """
    Returns sorted list of unique venues.
    """
    return sorted(matches_df["venue"].dropna().unique().astype(str).tolist())

def get_unique_cities(matches_df: pd.DataFrame) -> List[str]:
    """
    Returns sorted list of unique cities.
    """
    return sorted(matches_df["city"].dropna().unique().astype(str).tolist())

def filter_matches(
    matches_df: pd.DataFrame,
    season: str = "All",
    team: str = "All",
    venue: str = "All"
) -> pd.DataFrame:
    """
    Filters matches dataframe based on season, team, and venue selections.
    """
    df: pd.DataFrame = matches_df.copy()
    if season != "All":
        try:
            df = df[df["season"] == int(season)]
        except ValueError:
            pass
    if team != "All":
        df = df[(df["team1"] == team) | (df["team2"] == team)]
    if venue != "All":
        df = df[df["venue"] == venue]
    return df

def prepare_win_prediction_data(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the ball-by-ball deliveries dataset into a chase dataset 
    suitable for training a match win prediction model.
    Uses vectorized operations for production-grade speed.
    """
    # 1. Group deliveries to compute first innings total scores
    first_innings = deliveries_df[deliveries_df["inning"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
    first_innings.rename(columns={"total_runs": "target_runs"}, inplace=True)
    # The target runs for second innings is first innings total + 1
    first_innings["target_runs"] = first_innings["target_runs"] + 1
    
    # 2. Merge target score back to deliveries
    processed_deliveries = deliveries_df.merge(first_innings, on="match_id")
    
    # 3. Filter for second innings (the chase)
    chase_df = processed_deliveries[processed_deliveries["inning"] == 2].copy()
    
    # 4. Sort and calculate cumulative runs for each match
    chase_df.sort_values(by=["match_id", "over", "ball"], inplace=True)
    chase_df["current_score"] = chase_df.groupby("match_id")["total_runs"].cumsum()
    
    # 5. Runs left to win
    chase_df["runs_left"] = chase_df["target_runs"] - chase_df["current_score"]
    # If runs left goes negative (e.g. hit a boundary on the last ball), set to 0
    chase_df["runs_left"] = chase_df["runs_left"].clip(lower=0)
    
    # 6. Balls left (assuming standard 120-ball innings)
    chase_df["balls_left"] = 120 - ((chase_df["over"] - 1) * 6 + chase_df["ball"])
    chase_df["balls_left"] = chase_df["balls_left"].clip(lower=0, upper=120)
    
    # 7. Wickets left
    # Determine if delivery resulted in a wicket (Vectorized operation)
    dismissed_series = chase_df["player_dismissed"].astype(str).str.strip()
    chase_df["is_wicket"] = (~chase_df["player_dismissed"].isna() & 
                             ~dismissed_series.isin(["", "0", "nan", "None", "none"])).astype(int)
    
    chase_df["wickets_fallen"] = chase_df.groupby("match_id")["is_wicket"].cumsum()
    chase_df["wickets_left"] = 10 - chase_df["wickets_fallen"]
    chase_df["wickets_left"] = chase_df["wickets_left"].clip(lower=0, upper=10)
    
    # 8. Calculate Run Rates (CRR and RRR)
    balls_played = 120 - chase_df["balls_left"]
    chase_df["crr"] = np.where(balls_played > 0, (chase_df["current_score"] * 6) / balls_played, 0.0)
    chase_df["rrr"] = np.where(chase_df["balls_left"] > 0, (chase_df["runs_left"] * 6) / chase_df["balls_left"], 0.0)
    
    # 9. Merge with matches metadata (winner and venue)
    # City column is more consolidated than venue, we'll map city from matches.
    matches_meta = matches_df[["id", "winner", "city"]].copy()
    chase_df = chase_df.merge(matches_meta, left_on="match_id", right_on="id")
    
    # 10. Label target outcome: 1 if batting team won, 0 otherwise
    chase_df["result"] = np.where(chase_df["winner"] == chase_df["batting_team"], 1, 0)
    
    # 11. Select final columns for ML training
    final_features = [
        "batting_team",
        "bowling_team",
        "city",
        "runs_left",
        "balls_left",
        "wickets_left",
        "target_runs",
        "crr",
        "rrr",
        "result"
    ]
    
    # Filter rows: Remove cases where city is unknown to keep model training clean
    ml_df = chase_df[final_features].copy()
    ml_df.dropna(subset=["city", "batting_team", "bowling_team"], inplace=True)
    ml_df = ml_df[ml_df["city"] != "Unknown"]
    
    # Make sure we don't have infinite RRR or extreme outlier situations
    ml_df = ml_df[np.isfinite(ml_df["crr"]) & np.isfinite(ml_df["rrr"])]
    
    return ml_df
