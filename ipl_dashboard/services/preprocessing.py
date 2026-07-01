import pandas as pd
import numpy as np
from typing import List

def get_unique_seasons(matches_df: pd.DataFrame) -> List[int]:
    """Returns sorted list of unique seasons from matches."""
    if "season" not in matches_df.columns:
        return []
    return sorted(matches_df["season"].dropna().unique().astype(int).tolist(), reverse=True)

def get_unique_teams(matches_df: pd.DataFrame) -> List[str]:
    """Returns sorted list of unique teams from matches."""
    teams1 = set(matches_df["team1"].dropna().unique())
    teams2 = set(matches_df["team2"].dropna().unique())
    teams = teams1.union(teams2)
    teams = {t for t in teams if str(t).strip() not in ["", "No Result", "Unknown"]}
    return sorted(list(teams))

def get_unique_venues(matches_df: pd.DataFrame) -> List[str]:
    """Returns sorted list of unique venues."""
    return sorted(matches_df["venue"].dropna().unique().astype(str).tolist())

def get_unique_cities(matches_df: pd.DataFrame) -> List[str]:
    """Returns sorted list of unique cities."""
    return sorted(matches_df["city"].dropna().unique().astype(str).tolist())

def filter_matches(
    matches_df: pd.DataFrame,
    season: str = "All",
    team: str = "All",
    venue: str = "All"
) -> pd.DataFrame:
    """Filters matches dataframe based on season, team, and venue selections."""
    df = matches_df.copy()
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

def _get_first_innings_targets(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Computes targets from first innings total runs + 1."""
    first_inn = deliveries_df[deliveries_df["inning"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
    first_inn.rename(columns={"total_runs": "target_runs"}, inplace=True)
    first_inn["target_runs"] = first_inn["target_runs"] + 1
    return first_inn

def _calculate_chase_state(chase_df: pd.DataFrame) -> pd.DataFrame:
    """Computes running scores, wickets, balls, and runs left in second innings."""
    chase_df.sort_values(by=["match_id", "over", "ball"], inplace=True)
    chase_df["current_score"] = chase_df.groupby("match_id")["total_runs"].cumsum()
    chase_df["runs_left"] = (chase_df["target_runs"] - chase_df["current_score"]).clip(lower=0)
    chase_df["balls_left"] = (120 - ((chase_df["over"] - 1) * 6 + chase_df["ball"])).clip(lower=0, upper=120)
    
    # Calculate wickets left
    dismissed_series = chase_df["player_dismissed"].astype(str).str.strip()
    chase_df["is_wicket"] = (~chase_df["player_dismissed"].isna() & 
                             ~dismissed_series.isin(["", "0", "nan", "None", "none"])).astype(int)
    chase_df["wickets_fallen"] = chase_df.groupby("match_id")["is_wicket"].cumsum()
    chase_df["wickets_left"] = (10 - chase_df["wickets_fallen"]).clip(lower=0, upper=10)
    return chase_df

def _add_rates_and_labels(chase_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates run rates and joins matches metadata to label win outcomes."""
    balls_played = 120 - chase_df["balls_left"]
    chase_df["crr"] = np.where(balls_played > 0, (chase_df["current_score"] * 6) / balls_played, 0.0)
    chase_df["rrr"] = np.where(chase_df["balls_left"] > 0, (chase_df["runs_left"] * 6) / chase_df["balls_left"], 0.0)
    
    matches_meta = matches_df[["id", "winner", "city"]].copy()
    chase_df = chase_df.merge(matches_meta, left_on="match_id", right_on="id")
    chase_df["result"] = np.where(chase_df["winner"] == chase_df["batting_team"], 1, 0)
    return chase_df

def prepare_win_prediction_data(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw ball-by-ball datasets into chase dataset for ML training."""
    first_innings = _get_first_innings_targets(deliveries_df)
    processed_df = deliveries_df.merge(first_innings, on="match_id")
    
    chase_df = processed_df[processed_df["inning"] == 2].copy()
    chase_df = _calculate_chase_state(chase_df)
    chase_df = _add_rates_and_labels(chase_df, matches_df)
    
    features = ["batting_team", "bowling_team", "city", "runs_left", "balls_left", "wickets_left", "target_runs", "crr", "rrr", "result"]
    ml_df = chase_df[features].copy()
    ml_df.dropna(subset=["city", "batting_team", "bowling_team"], inplace=True)
    ml_df = ml_df[ml_df["city"] != "Unknown"]
    
    # Filter infinite rates
    ml_df = ml_df[np.isfinite(ml_df["crr"]) & np.isfinite(ml_df["rrr"])]
    return ml_df
