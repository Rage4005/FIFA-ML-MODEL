"""
Tournament Simulator Module — FIFA World Cup 2026 Prediction

Monte Carlo simulation of the full 48-team World Cup tournament.
Handles:
- Group stage (3 matches per team, points-based ranking with tiebreakers)
- Round of 32 (top 2 from each group + 8 best third-place teams)
- Round of 16, Quarter-finals, Semi-finals, Final
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json


def predict_match_proba(model, features_func, team_a, team_b, matches_df, 
                        rankings_df=None, elo_df=None, metadata_df=None,
                        match_date=None):
    """
    Predict match outcome probabilities using the trained model.
    
    Returns:
        dict with keys: team_a_win, draw, team_b_win (probabilities)
    """
    if match_date is None:
        match_date = pd.Timestamp.now()
    
    feats = features_func(
        matches_df, team_a, team_b, match_date,
        rankings_df, elo_df, metadata_df
    )
    
    X = pd.DataFrame([feats]).fillna(0)
    
    # Ensure columns match model's expected features
    if hasattr(model, 'get_booster'):
        model_features = model.get_booster().feature_names
        if model_features:
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[model_features]
    
    proba = model.predict_proba(X)[0]
    
    return {
        "team_a_win": float(proba[0]),
        "draw": float(proba[1]),
        "team_b_win": float(proba[2]),
    }


def simulate_match(proba, allow_draw=True):
    """
    Simulate a single match outcome based on probabilities.
    
    Returns: 'team_a', 'draw', or 'team_b'
    """
    if not allow_draw:
        # Redistribute draw probability for knockout matches
        total = proba["team_a_win"] + proba["team_b_win"]
        if total == 0:
            total = 1
        p_a = proba["team_a_win"] / total
        roll = np.random.random()
        return "team_a" if roll < p_a else "team_b"
    
    roll = np.random.random()
    if roll < proba["team_a_win"]:
        return "team_a"
    elif roll < proba["team_a_win"] + proba["draw"]:
        return "draw"
    else:
        return "team_b"


def simulate_group_stage(groups, model, features_func, matches_df,
                         rankings_df=None, elo_df=None, metadata_df=None):
    """
    Simulate the entire group stage.
    
    Each group has 4 teams playing 3 matches each (round-robin).
    Returns standings for each group sorted by: points, goal_diff, goals_scored.
    """
    from itertools import combinations
    
    group_standings = {}
    
    for group_name, teams in groups.items():
        standings = {team: {"points": 0, "gf": 0, "ga": 0, "gd": 0, "wins": 0, "draws": 0, "losses": 0}
                     for team in teams}
        
        # Round-robin: each pair plays once
        for team_a, team_b in combinations(teams, 2):
            proba = predict_match_proba(
                model, features_func, team_a, team_b, matches_df,
                rankings_df, elo_df, metadata_df
            )
            
            result = simulate_match(proba, allow_draw=True)
            
            # Simulate approximate scoreline
            if result == "team_a":
                goals_a = np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
                goals_b = np.random.choice([0, 1], p=[0.55, 0.45])
                standings[team_a]["points"] += 3
                standings[team_a]["wins"] += 1
                standings[team_b]["losses"] += 1
            elif result == "team_b":
                goals_b = np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
                goals_a = np.random.choice([0, 1], p=[0.55, 0.45])
                standings[team_b]["points"] += 3
                standings[team_b]["wins"] += 1
                standings[team_a]["losses"] += 1
            else:  # draw
                goals_a = goals_b = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])
                standings[team_a]["points"] += 1
                standings[team_b]["points"] += 1
                standings[team_a]["draws"] += 1
                standings[team_b]["draws"] += 1
            
            standings[team_a]["gf"] += goals_a
            standings[team_a]["ga"] += goals_b
            standings[team_a]["gd"] += goals_a - goals_b
            standings[team_b]["gf"] += goals_b
            standings[team_b]["ga"] += goals_a
            standings[team_b]["gd"] += goals_b - goals_a
        
        # Sort: points → goal difference → goals scored
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
            reverse=True
        )
        
        group_standings[group_name] = sorted_teams
    
    return group_standings


def get_knockout_teams(group_standings):
    """
    Determine which teams advance from group stage.
    
    2026 Format (48 teams, 12 groups):
    - Top 2 from each group (24 teams)
    - 8 best third-place teams (from 12 groups)
    - Total: 32 teams advance to knockout
    """
    advancing = []
    third_place = []
    
    for group_name, standings in group_standings.items():
        # Top 2 advance directly
        advancing.append(standings[0][0])  # 1st place
        advancing.append(standings[1][0])  # 2nd place
        
        # Collect 3rd place teams
        if len(standings) > 2:
            team_name = standings[2][0]
            team_stats = standings[2][1]
            third_place.append((team_name, team_stats, group_name))
    
    # Sort third-place teams: points → GD → GF
    third_place.sort(
        key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
        reverse=True
    )
    
    # Best 8 third-place teams advance
    for team_name, _, _ in third_place[:8]:
        advancing.append(team_name)
    
    return advancing


def simulate_knockout_round(teams, model, features_func, matches_df,
                            rankings_df=None, elo_df=None, metadata_df=None):
    """
    Simulate a single knockout round.
    
    Pairs teams sequentially and simulates each match (no draws allowed).
    Returns list of winners.
    """
    winners = []
    
    # Pair teams: 0v1, 2v3, 4v5, etc.
    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            # Odd number of teams, bye
            winners.append(teams[i])
            continue
        
        team_a = teams[i]
        team_b = teams[i + 1]
        
        proba = predict_match_proba(
            model, features_func, team_a, team_b, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        result = simulate_match(proba, allow_draw=False)
        
        if result == "team_a":
            winners.append(team_a)
        else:
            winners.append(team_b)
    
    return winners


def run_full_tournament(model, features_func, matches_df, groups,
                        rankings_df=None, elo_df=None, metadata_df=None,
                        n_simulations=10000, verbose=True):
    """
    Run Monte Carlo simulation of the entire World Cup tournament.
    
    Args:
        model: Trained match outcome classifier
        features_func: Function to build match features
        matches_df: Historical match data
        groups: Dict of group_name -> list of teams
        n_simulations: Number of tournament simulations
        verbose: Print progress
    
    Returns:
        Dict[team_name, win_count] → probability = win_count / n_simulations
    """
    win_counts = defaultdict(int)
    final_counts = defaultdict(int)  # How many times each team reaches the final
    semifinal_counts = defaultdict(int)  # How many times each team reaches semis
    
    for sim in range(n_simulations):
        if verbose and sim % 1000 == 0:
            print(f"   🏟️  Simulation {sim}/{n_simulations}...")
        
        # 1. Group Stage
        group_standings = simulate_group_stage(
            groups, model, features_func, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        # 2. Get 32 advancing teams
        knockout_teams = get_knockout_teams(group_standings)
        np.random.shuffle(knockout_teams)  # Randomize bracket seeding slightly
        
        # 3. Round of 32
        teams_16 = simulate_knockout_round(
            knockout_teams, model, features_func, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        # 4. Round of 16
        teams_8 = simulate_knockout_round(
            teams_16, model, features_func, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        # 5. Quarter-Finals
        teams_4 = simulate_knockout_round(
            teams_8, model, features_func, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        for team in teams_4:
            semifinal_counts[team] += 1
        
        # 6. Semi-Finals
        teams_2 = simulate_knockout_round(
            teams_4, model, features_func, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        for team in teams_2:
            final_counts[team] += 1
        
        # 7. Final
        finalists = simulate_knockout_round(
            teams_2, model, features_func, matches_df,
            rankings_df, elo_df, metadata_df
        )
        
        winner = finalists[0]
        win_counts[winner] += 1
    
    # Convert to probabilities
    results = {}
    all_teams = set()
    for group_teams in groups.values():
        all_teams.update(group_teams)
    
    for team in all_teams:
        results[team] = {
            "win_probability": round(win_counts[team] / n_simulations * 100, 2),
            "final_probability": round(final_counts.get(team, 0) / n_simulations * 100, 2),
            "semifinal_probability": round(semifinal_counts.get(team, 0) / n_simulations * 100, 2),
            "wins": win_counts[team],
            "finals": final_counts.get(team, 0),
            "semifinals": semifinal_counts.get(team, 0),
        }
    
    # Sort by win probability
    results = dict(sorted(results.items(), key=lambda x: x[1]["win_probability"], reverse=True))
    
    if verbose:
        print(f"\n🏆 Tournament Simulation Results ({n_simulations} simulations):")
        print(f"{'Team':<25} {'Win %':>8} {'Final %':>8} {'Semi %':>8}")
        print("-" * 55)
        for i, (team, stats) in enumerate(results.items()):
            if i < 15 or stats["win_probability"] > 1:
                print(f"{team:<25} {stats['win_probability']:>7.1f}% {stats['final_probability']:>7.1f}% {stats['semifinal_probability']:>7.1f}%")
    
    return results


def quick_simulate(model, features_func, matches_df, groups,
                   rankings_df=None, elo_df=None, metadata_df=None,
                   n_simulations=1000):
    """
    Quick simulation with fewer iterations for dashboard responsiveness.
    """
    return run_full_tournament(
        model, features_func, matches_df, groups,
        rankings_df, elo_df, metadata_df,
        n_simulations=n_simulations, verbose=False
    )
