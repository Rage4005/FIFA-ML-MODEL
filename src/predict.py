"""
Prediction API Module — FIFA World Cup 2026

High-level API for making predictions:
- Single match prediction
- Team win probability
- Side-by-side team comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path

from . import data_loader
from . import features
from . import model as model_module


def predict_match(team_a, team_b, model=None, verbose=True):
    """
    Predict the outcome of a match between team_a and team_b.
    
    Returns dict with:
    - probabilities (team_a_win, draw, team_b_win)
    - predicted winner
    - confidence level
    - key stats comparison
    """
    # Load model if not provided
    if model is None:
        model, metadata = model_module.load_model("xgb_match_predictor")
    
    # Load data
    matches_df = data_loader.load_matches()
    rankings_df = data_loader.load_rankings()
    elo_df = data_loader.load_elo_ratings()
    metadata_df = data_loader.load_team_metadata()
    
    # Build features
    match_date = pd.Timestamp.now()
    feats = features.build_match_features(
        matches_df, team_a, team_b, match_date,
        rankings_df, elo_df, metadata_df
    )
    
    X = pd.DataFrame([feats]).fillna(0)
    
    # Ensure feature alignment with model
    if hasattr(model, 'get_booster'):
        model_features = model.get_booster().feature_names
        if model_features:
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[model_features]
    
    # Predict
    proba = model.predict_proba(X)[0]
    
    # Determine winner
    outcome_map = {0: team_a, 1: "Draw", 2: team_b}
    predicted_idx = np.argmax(proba)
    predicted_winner = outcome_map[predicted_idx]
    confidence = float(proba[predicted_idx])
    
    result = {
        "team_a": team_a,
        "team_b": team_b,
        "probabilities": {
            f"{team_a}_win": round(float(proba[0]) * 100, 1),
            "draw": round(float(proba[1]) * 100, 1),
            f"{team_b}_win": round(float(proba[2]) * 100, 1),
        },
        "predicted_winner": predicted_winner,
        "confidence": round(confidence * 100, 1),
        "features_used": len(feats),
    }
    
    # Add team comparison
    result["comparison"] = compare_teams(team_a, team_b, matches_df, metadata_df, elo_df)
    
    if verbose:
        print(f"\n⚽ {team_a} vs {team_b}")
        print(f"{'='*40}")
        print(f"   {team_a} Win:  {result['probabilities'][f'{team_a}_win']}%")
        print(f"   Draw:         {result['probabilities']['draw']}%")
        print(f"   {team_b} Win:  {result['probabilities'][f'{team_b}_win']}%")
        print(f"\n   🏆 Predicted Winner: {predicted_winner} ({result['confidence']}% confidence)")
    
    return result


def compare_teams(team_a, team_b, matches_df=None, metadata_df=None, elo_df=None):
    """
    Generate side-by-side comparison of two teams.
    
    Returns structured comparison dict for dashboard display.
    """
    if matches_df is None:
        matches_df = data_loader.load_matches()
    if metadata_df is None:
        metadata_df = data_loader.load_team_metadata()
    if elo_df is None:
        elo_df = data_loader.load_elo_ratings()
    
    comparison = {}
    
    # Team metadata
    meta_a = metadata_df[metadata_df["team"] == team_a]
    meta_b = metadata_df[metadata_df["team"] == team_b]
    
    if len(meta_a) > 0 and len(meta_b) > 0:
        ma = meta_a.iloc[0]
        mb = meta_b.iloc[0]
        
        comparison["fifa_ranking"] = {team_a: int(ma["fifa_rank"]), team_b: int(mb["fifa_rank"])}
        comparison["elo_rating"] = {team_a: int(ma["elo_rating"]), team_b: int(mb["elo_rating"])}
        comparison["market_value"] = {team_a: float(ma["market_value_billion_eur"]), team_b: float(mb["market_value_billion_eur"])}
        comparison["avg_squad_age"] = {team_a: float(ma["avg_squad_age"]), team_b: float(mb["avg_squad_age"])}
        comparison["top5_league_players"] = {team_a: int(ma["top5_league_players"]), team_b: int(mb["top5_league_players"])}
        comparison["wc_appearances"] = {team_a: int(ma["wc_appearances"]), team_b: int(mb["wc_appearances"])}
        comparison["wc_best_finish"] = {team_a: ma["wc_best_finish"], team_b: mb["wc_best_finish"]}
        comparison["strength_tier"] = {team_a: int(ma["strength_tier"]), team_b: int(mb["strength_tier"])}
    
    # Recent form
    match_date = pd.Timestamp.now()
    form_a_5 = features.compute_team_rolling_stats(matches_df, team_a, match_date, window=5)
    form_b_5 = features.compute_team_rolling_stats(matches_df, team_b, match_date, window=5)
    
    comparison["last_5_wins"] = {team_a: form_a_5["wins_last_5"], team_b: form_b_5["wins_last_5"]}
    comparison["last_5_goals_scored"] = {team_a: form_a_5["goals_scored_last_5"], team_b: form_b_5["goals_scored_last_5"]}
    comparison["last_5_goals_conceded"] = {team_a: form_a_5["goals_conceded_last_5"], team_b: form_b_5["goals_conceded_last_5"]}
    comparison["last_5_clean_sheets"] = {team_a: form_a_5["clean_sheets_last_5"], team_b: form_b_5["clean_sheets_last_5"]}
    
    # Head to head
    h2h = data_loader.get_head_to_head(matches_df, team_a, team_b)
    comparison["head_to_head"] = h2h
    
    return comparison


def get_team_profile(team_name):
    """
    Get comprehensive profile for a single team.
    
    Returns all stats, form, and tournament history for dashboard display.
    """
    matches_df = data_loader.load_matches()
    metadata_df = data_loader.load_team_metadata()
    elo_df = data_loader.load_elo_ratings()
    injuries = data_loader.load_injuries()
    
    meta = metadata_df[metadata_df["team"] == team_name]
    if len(meta) == 0:
        return {"error": f"Team '{team_name}' not found"}
    
    meta = meta.iloc[0]
    match_date = pd.Timestamp.now()
    
    # Form stats
    form_5 = features.compute_team_rolling_stats(matches_df, team_name, match_date, 5)
    form_10 = features.compute_team_rolling_stats(matches_df, team_name, match_date, 10)
    
    # WC history
    wc_stats = features.compute_wc_historical_features(matches_df, team_name)
    
    # Injury info
    team_injuries = injuries.get(team_name, {"injured_key_players": 0, "details": []})
    
    # Radar chart data (normalized 0-100)
    radar = compute_radar_stats(meta, form_5, form_10, wc_stats, metadata_df)
    
    profile = {
        "team": team_name,
        "group": meta["group"],
        "confederation": meta["confederation"],
        "fifa_rank": int(meta["fifa_rank"]),
        "elo_rating": int(meta["elo_rating"]),
        "market_value_b": float(meta["market_value_billion_eur"]),
        "avg_squad_age": float(meta["avg_squad_age"]),
        "top5_league_players": int(meta["top5_league_players"]),
        "wc_appearances": int(meta["wc_appearances"]),
        "wc_best_finish": meta["wc_best_finish"],
        "strength_tier": int(meta["strength_tier"]),
        "form_5": form_5,
        "form_10": form_10,
        "wc_stats": wc_stats,
        "injuries": team_injuries,
        "radar": radar,
    }
    
    return profile


def compute_radar_stats(meta, form_5, form_10, wc_stats, metadata_df):
    """
    Compute normalized radar chart stats (0-100 scale).
    Categories: Attack, Defense, Form, Experience, Squad Quality
    """
    # Normalize against all teams' ranges
    max_elo = metadata_df["elo_rating"].max()
    min_elo = metadata_df["elo_rating"].min()
    max_mv = metadata_df["market_value_billion_eur"].max()
    max_top5 = metadata_df["top5_league_players"].max()
    max_wc = metadata_df["wc_appearances"].max()
    
    elo_norm = (meta["elo_rating"] - min_elo) / (max_elo - min_elo) * 100 if max_elo > min_elo else 50
    
    # Attack: based on goals scored and offensive form
    attack = min(100, (form_5.get("avg_goals_scored_last_5", 0) / 3.0) * 100)
    
    # Defense: based on goals conceded (inverted - fewer = better)
    defense = max(0, 100 - (form_5.get("avg_goals_conceded_last_5", 1.5) / 3.0) * 100)
    
    # Form: based on win rate
    form = form_10.get("win_rate_last_10", 0) * 100
    
    # Experience: WC appearances and best finish
    experience = min(100, (meta["wc_appearances"] / max_wc * 50) + (meta["wc_best_finish_code"] / 7.0 * 50))
    
    # Squad Quality: market value + top 5 league players
    squad = min(100, (meta["market_value_billion_eur"] / max_mv * 60) + (meta["top5_league_players"] / max_top5 * 40))
    
    return {
        "Attack": round(attack, 1),
        "Defense": round(defense, 1),
        "Form": round(form, 1),
        "Experience": round(experience, 1),
        "Squad Quality": round(squad, 1),
        "Overall": round(elo_norm, 1),
    }
