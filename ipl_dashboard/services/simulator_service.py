import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def simulate_innings(batting_team: str, bowling_team: str, venue_stats: Dict[str, Any], target: int = None) -> List[Dict[str, Any]]:
    """
    Simulates a 20-over innings using historical phase statistics and random distributions.
    If target is provided, it simulates a run chase and terminates if the target is exceeded.
    """
    runs = 0
    wickets = 0
    timeline = [{"Over": 0, "Runs": 0, "Wickets": 0}]
    
    # Establish base run-rates based on venue indicators
    pp_rr = venue_stats.get("avg_powerplay", 45.0) / 6.0
    death_rr = venue_stats.get("avg_death", 40.0) / 5.0
    mid_rr = (venue_stats.get("avg_1st_innings", 160.0) - (pp_rr * 6 + death_rr * 5)) / 9.0
    mid_rr = max(6.0, min(10.0, mid_rr))
    
    for over in range(1, 21):
        if wickets >= 10:
            break
            
        # Determine run rate and wicket chance for the phase
        if over <= 6:
            mean_rr, wkt_prob = pp_rr, 0.08
        elif over <= 15:
            mean_rr, wkt_prob = mid_rr, 0.07
        else:
            mean_rr, wkt_prob = death_rr, 0.18
            
        # Add random variation to runs and wickets
        over_runs = int(round(np.random.normal(mean_rr, 1.8)))
        over_runs = max(0, over_runs)
        
        over_wkts = np.random.binomial(2, wkt_prob)
        over_wkts = min(10 - wickets, over_wkts)
        
        runs += over_runs
        wickets += over_wkts
        
        timeline.append({"Over": over, "Runs": runs, "Wickets": wickets})
        
        if target is not None and runs >= target:
            break
            
    return timeline

def predict_first_innings_score(
    batting_team: str,
    bowling_team: str,
    venue: str,
    current_runs: int,
    wickets: int,
    overs: float,
    avg_venue_score: float
) -> int:
    """
    Predicts the first innings score using a weighted projected run-rate formula.
    Integrates venue metrics, current run-rate, and remaining wickets resource.
    """
    ov = int(overs)
    balls = int(round((overs - ov) * 10))
    balls_played = ov * 6 + balls
    balls_left = max(0, 120 - balls_played)
    
    crr = (current_runs * 6) / balls_played if balls_played > 0 else 7.5
    
    # Wicket resource weight: more wickets remaining means team can accelerate
    wkt_resource = (10 - wickets) / 10.0
    
    # Baseline expected run rate for the remaining overs
    base_proj_rr = (avg_venue_score / 20.0) if avg_venue_score > 0 else 8.0
    
    # Projected run rate is a weighted blend
    proj_rr = (crr * 0.4) + (base_proj_rr * 0.6 * (0.7 + 0.3 * wkt_resource))
    
    projected_runs = int(round(current_runs + (proj_rr * balls_left / 6)))
    return max(current_runs, projected_runs)

def _simulate_head_to_head(team_a: str, team_b: str, h2h_map: Dict[Tuple[str, str], float]) -> str:
    """Helper to simulate single match outcome based on team win probability."""
    prob = h2h_map.get((team_a, team_b), 0.5)
    return team_a if np.random.random() < prob else team_b

def simulate_season(teams: List[str], team_win_probs: Dict[Tuple[str, str], float]) -> Dict[str, int]:
    """
    Simulates a full season tournament (round robin + standard IPL playoffs).
    Returns a dictionary of total tournament wins for each team.
    """
    points = {t: 0 for t in teams}
    
    # 1. Round Robin: Each team plays everyone else once
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            winner = _simulate_head_to_head(t1, t2, team_win_probs)
            points[winner] += 2
            
    # Sort standings
    standings = sorted(points.items(), key=lambda x: x[1], reverse=True)
    playoff_teams = [t[0] for t in standings[:4]]
    if len(playoff_teams) < 4:
         return {t: 0 for t in teams}
         
    # 2. Playoffs simulation
    # Qualifier 1: 1st vs 2nd
    q1_winner = _simulate_head_to_head(playoff_teams[0], playoff_teams[1], team_win_probs)
    q1_loser = playoff_teams[1] if q1_winner == playoff_teams[0] else playoff_teams[0]
    
    # Eliminator: 3rd vs 4th
    elim_winner = _simulate_head_to_head(playoff_teams[2], playoff_teams[3], team_win_probs)
    
    # Qualifier 2: q1_loser vs elim_winner
    q2_winner = _simulate_head_to_head(q1_loser, elim_winner, team_win_probs)
    
    # Final: q1_winner vs q2_winner
    champion = _simulate_head_to_head(q1_winner, q2_winner, team_win_probs)
    
    return {champion: 1}
