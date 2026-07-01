import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Dict
from constants.paths import MATCHES_PATH, DELIVERIES_PATH
from constants.teams import TEAM_NAME_MAP

DATA_URLS: Dict[str, List[str]] = {
    "matches": [
        "https://raw.githubusercontent.com/lawalsegun2025/ipl_match_win_predictor/main/matches.csv",
        "https://raw.githubusercontent.com/Shivaae/IPL-DATA-/master/matches.csv",
        "https://raw.githubusercontent.com/srinathkr07/IPL-Data-Analysis/master/matches.csv"
    ],
    "deliveries": [
        "https://raw.githubusercontent.com/lawalsegun2025/ipl_match_win_predictor/main/deliveries.csv",
        "https://raw.githubusercontent.com/Shivaae/IPL-DATA-/master/deliveries.csv",
        "https://raw.githubusercontent.com/srinathkr07/IPL-Data-Analysis/master/deliveries.csv"
    ]
}

def download_file(urls: List[str], dest_path: str, file_type: str) -> bool:
    """Downloads a file from a list of fallback URLs to destination."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        status_placeholder = st.empty()
        status_placeholder.info(f"Downloading {file_type} dataset. This might take a few moments...")
    except Exception:
        status_placeholder = None
        print(f"Downloading {file_type} dataset...")

    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if status_placeholder:
                    status_placeholder.empty()
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue

    if status_placeholder:
        status_placeholder.error(f"Failed to download {file_type} dataset.")
    return False

def check_and_download_data() -> bool:
    """Checks if matches and deliveries datasets exist, else downloads them."""
    success = True
    if not os.path.exists(MATCHES_PATH):
        success = success and download_file(DATA_URLS["matches"], MATCHES_PATH, "Matches")
    if not os.path.exists(DELIVERIES_PATH):
        success = success and download_file(DATA_URLS["deliveries"], DELIVERIES_PATH, "Deliveries")
    return success

def standardize_team_names(df: pd.DataFrame, team_cols: List[str]) -> pd.DataFrame:
    """Standardizes team names in specified columns of the dataframe."""
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].replace(TEAM_NAME_MAP)
            df[col] = df[col].astype(str).str.strip()
    return df

def _parse_season(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to extract season year from date or season column."""
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    elif "date" in df.columns:
        df["season"] = df["date"].astype(str).str.extract(r"(\d{4})")[0].fillna(0).astype(float).astype(int)
    return df

@st.cache_data(show_spinner="Loading and cleaning IPL Match Data...")
def load_matches_data() -> pd.DataFrame:
    """Loads matches.csv and cleans columns/values."""
    check_and_download_data()
    if not os.path.exists(MATCHES_PATH):
        raise FileNotFoundError(f"Matches dataset not found at {MATCHES_PATH}")
    df = pd.read_csv(MATCHES_PATH)
    df.columns = [col.lower() for col in df.columns]
    df = standardize_team_names(df, ["team1", "team2", "toss_winner", "winner"])
    df["winner"] = df["winner"].fillna("No Result")
    df["city"] = df["city"].fillna("Unknown").replace({"Bengaluru": "Bangalore"})
    df["venue"] = df["venue"].fillna("Unknown")
    df = _parse_season(df)
    
    # Map win_by_runs/win_by_wickets if they are missing
    if "win_by_runs" not in df.columns and "result" in df.columns and "result_margin" in df.columns:
        df["result_margin"] = pd.to_numeric(df["result_margin"], errors="coerce").fillna(0).astype(int)
        df["win_by_runs"] = np.where(df["result"] == "runs", df["result_margin"], 0)
        df["win_by_wickets"] = np.where(df["result"] == "wickets", df["result_margin"], 0)
    
    if "win_by_runs" in df.columns:
        df["win_by_runs"] = pd.to_numeric(df["win_by_runs"], errors="coerce").fillna(0).astype(int)
    if "win_by_wickets" in df.columns:
        df["win_by_wickets"] = pd.to_numeric(df["win_by_wickets"], errors="coerce").fillna(0).astype(int)
        
    return df

@st.cache_data(show_spinner="Loading and cleaning IPL Ball-by-Ball Data...")
def load_deliveries_data() -> pd.DataFrame:
    """Loads deliveries.csv and cleans columns/values."""
    check_and_download_data()
    if not os.path.exists(DELIVERIES_PATH):
        raise FileNotFoundError(f"Deliveries dataset not found at {DELIVERIES_PATH}")
    df = pd.read_csv(DELIVERIES_PATH)
    df.columns = [col.lower() for col in df.columns]
    if "id" in df.columns and "match_id" not in df.columns:
        df = df.rename(columns={"id": "match_id"})
    df = standardize_team_names(df, ["batting_team", "bowling_team"])
    
    run_cols = ["batsman_runs", "extra_runs", "total_runs", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"]
    for col in run_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df

def get_combined_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns loaded matches and deliveries datasets."""
    return load_matches_data(), load_deliveries_data()
