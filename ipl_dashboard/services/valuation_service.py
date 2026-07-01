import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List, Tuple

def find_similar_players(player_name: str, stats_df: pd.DataFrame, player_type: str = "batsman", limit: int = 5) -> List[Tuple[str, float]]:
    """
    Finds top N similar players based on Cosine Similarity of normalized performance stats.
    """
    cols = ["Runs", "Average", "Strike Rate", "Innings"] if player_type == "batsman" else ["Wickets", "Economy", "Average", "Innings"]
    
    work_df = stats_df.copy().dropna(subset=["Player"])
    if player_name not in work_df["Player"].values:
        return []
        
    for col in cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce").fillna(0.0)
        
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(work_df[cols])
    
    player_idx = work_df[work_df["Player"] == player_name].index[0]
    player_vec = scaled[work_df.index.get_loc(player_idx)]
    
    similarities = []
    for i, row_idx in enumerate(work_df.index):
        if row_idx == player_idx:
            continue
        vec = scaled[i]
        
        # Calculate Cosine Similarity
        dot_product = np.dot(player_vec, vec)
        norm_a = np.linalg.norm(player_vec)
        norm_b = np.linalg.norm(vec)
        
        sim = (dot_product / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
        similarities.append((work_df.loc[row_idx, "Player"], float(round(sim * 100, 1))))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:limit]

def estimate_auction_value(player_name: str, batting_stats: pd.DataFrame, bowling_stats: pd.DataFrame) -> Tuple[float, str]:
    """
    Estimates the auction value of a player in Crores (INR).
    Returns a tuple: (estimated_value_in_crores, salary_tier_name).
    """
    # 1. Check Batting
    bat_row = batting_stats[batting_stats["Player"] == player_name]
    runs = bat_row["Runs"].values[0] if not bat_row.empty else 0
    avg = bat_row["Average"].values[0] if not bat_row.empty else 0
    sr = bat_row["Strike Rate"].values[0] if not bat_row.empty else 0
    
    # 2. Check Bowling
    bowl_row = bowling_stats[bowling_stats["Player"] == player_name]
    wkts = bowl_row["Wickets"].values[0] if not bowl_row.empty else 0
    econ = bowl_row["Economy"].values[0] if not bowl_row.empty else 9.0
    
    # Calculate performance index (0 to 100)
    bat_index = (runs * 0.01) + (avg * 0.4) + ((sr - 100) * 0.2 if sr > 100 else 0)
    bowl_index = (wkts * 0.8) + ((10 - econ) * 5.0 if econ < 10 else 0)
    
    overall_index = max(0, max(bat_index, bowl_index) + min(bat_index, bowl_index) * 0.5)
    overall_index = min(100.0, overall_index)
    
    # Map index to auction price (Base price: 20L up to 20Cr)
    if overall_index >= 75.0:
        val = 12.0 + (overall_index - 75.0) / 25.0 * 12.0
        tier = "Marquee / Icon Player"
    elif overall_index >= 50.0:
        val = 5.0 + (overall_index - 50.0) / 25.0 * 7.0
        tier = "Premium Performer"
    elif overall_index >= 25.0:
        val = 1.5 + (overall_index - 25.0) / 25.0 * 3.5
        tier = "Consistent Regular"
    else:
        val = 0.2 + (overall_index / 25.0) * 1.3
        tier = "Squad Player / Base Price"
        
    return float(round(val, 2)), tier
