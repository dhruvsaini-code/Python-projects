import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, Any, Tuple

def get_player_pca_clustering(df: pd.DataFrame, player_type: str = "batsman", n_clusters: int = 4) -> pd.DataFrame:
    """
    Standardizes player stats, applies PCA (2 components), and runs K-Means clustering.
    Returns a copy of the dataframe with 'PCA1', 'PCA2', and 'Cluster' columns.
    """
    cols = ["Runs", "Average", "Strike Rate", "Innings"] if player_type == "batsman" else ["Wickets", "Economy", "Average", "Innings"]
    
    # Filter valid rows and fill NaNs
    work_df = df.copy().dropna(subset=["Player"])
    for col in cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce").fillna(0.0)
        
    X = work_df[cols]
    if len(X) < n_clusters:
        work_df["PCA1"] = 0.0
        work_df["PCA2"] = 0.0
        work_df["Cluster"] = 0
        return work_df
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    work_df["PCA1"] = X_pca[:, 0]
    work_df["PCA2"] = X_pca[:, 1]
    work_df["Cluster"] = clusters.astype(str)
    
    # Assign names to clusters for better UI representation
    if player_type == "batsman":
        cluster_names = {"0": "Anchors", "1": "Finishers/Power Hitters", "2": "Top Order Accumulators", "3": "Lower Order/Tail"}
    else:
        cluster_names = {"0": "Powerplay Strike Bowlers", "1": "Death Specialist Bowlers", "2": "Containing/Economy Spinners", "3": "Part-timers/Tail Bowlers"}
    work_df["Cluster Role"] = work_df["Cluster"].map(cluster_names).fillna("Unclassified")
    
    return work_df

def get_correlation_matrix(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Computes a correlation matrix of numerical metrics for match outcomes."""
    df = matches_df.copy()
    
    # Encode key features to numbers for correlation
    df["toss_decision_num"] = np.where(df["toss_decision"] == "field", 1, 0)
    df["win_by_runs"] = pd.to_numeric(df["win_by_runs"], errors="coerce").fillna(0)
    df["win_by_wickets"] = pd.to_numeric(df["win_by_wickets"], errors="coerce").fillna(0)
    df["toss_win_match_win"] = np.where(df["toss_winner"] == df["winner"], 1, 0)
    
    corr_cols = ["season", "toss_decision_num", "win_by_runs", "win_by_wickets", "toss_win_match_win"]
    return df[corr_cols].corr()

def get_team_similarity_matrix(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a similarity matrix between teams based on normalized success metrics.
    Returns a DataFrame representing pairwise cosine similarities.
    """
    from services.preprocessing import get_unique_teams
    teams = get_unique_teams(matches_df)
    
    stats = []
    for team in teams:
        t_matches = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)]
        wins = len(t_matches[matches_df["winner"] == team])
        win_rate = wins / len(t_matches) if len(t_matches) > 0 else 0
        toss_wins = len(t_matches[matches_df["toss_winner"] == team])
        toss_rate = toss_wins / len(t_matches) if len(t_matches) > 0 else 0
        
        # Avg win margin by runs / wickets
        avg_run_margin = t_matches[t_matches["winner"] == team]["win_by_runs"].mean()
        avg_wkt_margin = t_matches[t_matches["winner"] == team]["win_by_wickets"].mean()
        
        stats.append({
            "Team": team,
            "Win Rate": win_rate,
            "Toss Win Rate": toss_rate,
            "Avg Win Run Margin": 0.0 if pd.isna(avg_run_margin) else avg_run_margin,
            "Avg Win Wicket Margin": 0.0 if pd.isna(avg_wkt_margin) else avg_wkt_margin
        })
        
    df_stats = pd.DataFrame(stats).set_index("Team")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_stats)
    
    # pairwise cosine similarity (dot product of normalized vectors)
    norms = np.linalg.norm(scaled_features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = scaled_features / norms
    similarity = np.dot(normalized, normalized.T)
    
    return pd.DataFrame(similarity, index=df_stats.index, columns=df_stats.index)
