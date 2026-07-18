"""
Feature Engineering Module — FIFA World Cup 2026 Prediction

Builds all ML-ready features from raw data:
- Team strength (Elo, FIFA ranking, market value)
- Recent form (rolling wins, goals, clean sheets)
- Attacking & defensive performance
- Tournament context
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_team_rolling_stats(matches_df, team, before_date, window=5):
    """
    Compute rolling statistics for a team from their last N matches.
    
    Returns dict with:
    - wins, draws, losses, win_rate
    - goals_scored, goals_conceded, goal_difference
    - clean_sheets, avg_goals_scored, avg_goals_conceded
    - scoring_consistency (std dev), defensive_consistency (std dev)
    """
    # Get team's matches before the given date
    mask = (
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["date"] < before_date)
    )
    team_matches = matches_df[mask].sort_values("date", ascending=False).head(window)
    
    if len(team_matches) == 0:
        return _empty_rolling_stats(window)
    
    wins = 0
    draws = 0
    losses = 0
    goals_scored_list = []
    goals_conceded_list = []
    
    for _, match in team_matches.iterrows():
        if match["home_team"] == team:
            gs = match["home_score"]
            gc = match["away_score"]
        else:
            gs = match["away_score"]
            gc = match["home_score"]
        
        goals_scored_list.append(gs)
        goals_conceded_list.append(gc)
        
        if gs > gc:
            wins += 1
        elif gs == gc:
            draws += 1
        else:
            losses += 1
    
    n = len(team_matches)
    gs_arr = np.array(goals_scored_list)
    gc_arr = np.array(goals_conceded_list)
    
    return {
        f"wins_last_{window}": wins,
        f"draws_last_{window}": draws,
        f"losses_last_{window}": losses,
        f"win_rate_last_{window}": wins / n if n > 0 else 0,
        f"goals_scored_last_{window}": int(gs_arr.sum()),
        f"goals_conceded_last_{window}": int(gc_arr.sum()),
        f"goal_diff_last_{window}": int(gs_arr.sum() - gc_arr.sum()),
        f"clean_sheets_last_{window}": int((gc_arr == 0).sum()),
        f"avg_goals_scored_last_{window}": round(gs_arr.mean(), 2) if n > 0 else 0,
        f"avg_goals_conceded_last_{window}": round(gc_arr.mean(), 2) if n > 0 else 0,
        f"scoring_consistency_last_{window}": round(gs_arr.std(), 2) if n > 1 else 0,
        f"defensive_consistency_last_{window}": round(gc_arr.std(), 2) if n > 1 else 0,
        f"clean_sheet_pct_last_{window}": round((gc_arr == 0).sum() / n, 2) if n > 0 else 0,
        f"matches_played_last_{window}": n,
    }


def _empty_rolling_stats(window):
    """Return empty rolling stats when no matches are found."""
    return {
        f"wins_last_{window}": 0,
        f"draws_last_{window}": 0,
        f"losses_last_{window}": 0,
        f"win_rate_last_{window}": 0,
        f"goals_scored_last_{window}": 0,
        f"goals_conceded_last_{window}": 0,
        f"goal_diff_last_{window}": 0,
        f"clean_sheets_last_{window}": 0,
        f"avg_goals_scored_last_{window}": 0,
        f"avg_goals_conceded_last_{window}": 0,
        f"scoring_consistency_last_{window}": 0,
        f"defensive_consistency_last_{window}": 0,
        f"clean_sheet_pct_last_{window}": 0,
        f"matches_played_last_{window}": 0,
    }


def compute_vs_top_teams_stats(matches_df, team, before_date, rankings_df=None, top_n=20, window=10):
    """
    Compute performance against top-N ranked teams.
    Measures how a team performs against strong opposition.
    """
    mask = (
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["date"] < before_date)
    )
    team_matches = matches_df[mask].sort_values("date", ascending=False).head(window * 3)
    
    # If we have rankings, filter to matches against top teams
    # For simplicity, use a heuristic based on team metadata
    from . import data_loader
    try:
        metadata = data_loader.load_team_metadata()
        top_teams = set(metadata[metadata["fifa_rank"] <= top_n]["team"].tolist())
    except Exception:
        top_teams = set()
    
    wins_vs_top = 0
    matches_vs_top = 0
    goals_vs_top = 0
    
    for _, match in team_matches.iterrows():
        opponent = match["away_team"] if match["home_team"] == team else match["home_team"]
        
        if opponent in top_teams:
            matches_vs_top += 1
            if match["home_team"] == team:
                gs = match["home_score"]
                gc = match["away_score"]
            else:
                gs = match["away_score"]
                gc = match["home_score"]
            
            goals_vs_top += gs
            if gs > gc:
                wins_vs_top += 1
    
    return {
        "wins_vs_top20": wins_vs_top,
        "matches_vs_top20": matches_vs_top,
        "win_rate_vs_top20": round(wins_vs_top / matches_vs_top, 2) if matches_vs_top > 0 else 0,
        "avg_goals_vs_top20": round(goals_vs_top / matches_vs_top, 2) if matches_vs_top > 0 else 0,
    }


def compute_wc_historical_features(matches_df, team):
    """
    Compute World Cup historical performance features.
    """
    wc_matches = matches_df[
        (matches_df["tournament"] == "FIFA World Cup") &
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team))
    ]
    
    if len(wc_matches) == 0:
        return {
            "wc_total_matches": 0,
            "wc_total_wins": 0,
            "wc_win_rate": 0,
            "wc_total_goals": 0,
            "wc_avg_goals": 0,
        }
    
    wins = 0
    total_goals = 0
    
    for _, match in wc_matches.iterrows():
        if match["home_team"] == team:
            gs = match["home_score"]
            gc = match["away_score"]
        else:
            gs = match["away_score"]
            gc = match["home_score"]
        
        total_goals += gs
        if gs > gc:
            wins += 1
    
    n = len(wc_matches)
    return {
        "wc_total_matches": n,
        "wc_total_wins": wins,
        "wc_win_rate": round(wins / n, 2),
        "wc_total_goals": total_goals,
        "wc_avg_goals": round(total_goals / n, 2),
    }


def build_match_features(matches_df, home_team, away_team, match_date,
                         rankings_df=None, elo_df=None, metadata_df=None):
    """
    Build complete feature vector for a single match.
    
    Returns a dict of all features for the home_team vs away_team matchup.
    """
    features = {}
    
    # ===== Team Strength Features =====
    if metadata_df is not None:
        home_meta = metadata_df[metadata_df["team"] == home_team]
        away_meta = metadata_df[metadata_df["team"] == away_team]
        
        if len(home_meta) > 0:
            hm = home_meta.iloc[0]
            features["home_elo"] = hm["elo_rating"]
            features["home_fifa_rank"] = hm["fifa_rank"]
            features["home_market_value"] = hm["market_value_billion_eur"]
            features["home_avg_age"] = hm["avg_squad_age"]
            features["home_top5_players"] = hm["top5_league_players"]
            features["home_wc_appearances"] = hm["wc_appearances"]
            features["home_wc_best_code"] = hm["wc_best_finish_code"]
            features["home_strength_tier"] = hm["strength_tier"]
        
        if len(away_meta) > 0:
            am = away_meta.iloc[0]
            features["away_elo"] = am["elo_rating"]
            features["away_fifa_rank"] = am["fifa_rank"]
            features["away_market_value"] = am["market_value_billion_eur"]
            features["away_avg_age"] = am["avg_squad_age"]
            features["away_top5_players"] = am["top5_league_players"]
            features["away_wc_appearances"] = am["wc_appearances"]
            features["away_wc_best_code"] = am["wc_best_finish_code"]
            features["away_strength_tier"] = am["strength_tier"]
    
    # Elo from time-series data
    if elo_df is not None:
        from . import data_loader
        year = match_date.year if hasattr(match_date, 'year') else pd.Timestamp(match_date).year
        features["home_elo"] = data_loader.get_elo_at_year(elo_df, home_team, year)
        features["away_elo"] = data_loader.get_elo_at_year(elo_df, away_team, year)
    
    # FIFA Rankings
    if rankings_df is not None:
        from . import data_loader
        hr = data_loader.get_ranking_at_date(rankings_df, home_team, match_date)
        ar = data_loader.get_ranking_at_date(rankings_df, away_team, match_date)
        features["home_fifa_rank"] = hr["rank"]
        features["home_fifa_points"] = hr["total_points"]
        features["away_fifa_rank"] = ar["rank"]
        features["away_fifa_points"] = ar["total_points"]
    
    # ===== Differential Features =====
    if "home_elo" in features and "away_elo" in features:
        features["elo_diff"] = features["home_elo"] - features["away_elo"]
    if "home_fifa_rank" in features and "away_fifa_rank" in features:
        features["rank_diff"] = features["home_fifa_rank"] - features["away_fifa_rank"]
    if "home_market_value" in features and "away_market_value" in features:
        features["market_value_diff"] = features["home_market_value"] - features["away_market_value"]
    
    # ===== Recent Form Features =====
    for window in [5, 10]:
        home_form = compute_team_rolling_stats(matches_df, home_team, match_date, window)
        away_form = compute_team_rolling_stats(matches_df, away_team, match_date, window)
        
        for key, val in home_form.items():
            features[f"home_{key}"] = val
        for key, val in away_form.items():
            features[f"away_{key}"] = val
        
        # Form differentials
        features[f"form_diff_win_rate_{window}"] = (
            home_form.get(f"win_rate_last_{window}", 0) - 
            away_form.get(f"win_rate_last_{window}", 0)
        )
        features[f"form_diff_goals_{window}"] = (
            home_form.get(f"avg_goals_scored_last_{window}", 0) - 
            away_form.get(f"avg_goals_scored_last_{window}", 0)
        )
    
    # ===== World Cup History =====
    home_wc = compute_wc_historical_features(matches_df, home_team)
    away_wc = compute_wc_historical_features(matches_df, away_team)
    for key, val in home_wc.items():
        features[f"home_{key}"] = val
    for key, val in away_wc.items():
        features[f"away_{key}"] = val
    
    return features


def build_training_dataset(matches_df, rankings_df=None, elo_df=None, metadata_df=None,
                           start_year=2015, sample_size=None, verbose=True):
    """
    Build the full training dataset from historical matches.
    
    For each match, compute features for both teams and the outcome label.
    Uses matches from start_year onwards to ensure enough historical data for rolling stats.
    
    Returns (X, y) where X is a DataFrame of features and y is the outcome array.
    """
    # Filter matches
    df = matches_df[matches_df["year"] >= start_year].copy()
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Only include matches involving WC 2026 teams or teams with metadata
    if metadata_df is not None:
        known_teams = set(metadata_df["team"].tolist())
    else:
        known_teams = None
    
    rows = []
    outcomes = []
    skipped = 0
    
    total = len(df)
    for idx, (_, match) in enumerate(df.iterrows()):
        if verbose and idx % 500 == 0:
            print(f"   Processing match {idx+1}/{total}...")
        
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = match["date"]
        
        try:
            feats = build_match_features(
                matches_df, home_team, away_team, match_date,
                rankings_df, elo_df, metadata_df
            )
            
            if len(feats) < 5:
                skipped += 1
                continue
            
            rows.append(feats)
            outcomes.append(match["outcome"])
        except Exception as e:
            skipped += 1
            continue
    
    if verbose:
        print(f"   ✅ Built {len(rows)} feature vectors (skipped {skipped})")
    
    X = pd.DataFrame(rows)
    y = np.array(outcomes)
    
    # Fill NaN with 0 (missing features for teams without metadata)
    X = X.fillna(0)
    
    return X, y


def get_feature_names():
    """Return categorized feature names for interpretability."""
    return {
        "team_strength": [
            "home_elo", "away_elo", "elo_diff",
            "home_fifa_rank", "away_fifa_rank", "rank_diff",
            "home_fifa_points", "away_fifa_points",
            "home_market_value", "away_market_value", "market_value_diff",
            "home_strength_tier", "away_strength_tier",
        ],
        "recent_form_5": [
            "home_wins_last_5", "home_win_rate_last_5",
            "home_goals_scored_last_5", "home_goals_conceded_last_5",
            "home_goal_diff_last_5", "home_clean_sheets_last_5",
            "away_wins_last_5", "away_win_rate_last_5",
            "away_goals_scored_last_5", "away_goals_conceded_last_5",
            "away_goal_diff_last_5", "away_clean_sheets_last_5",
            "form_diff_win_rate_5", "form_diff_goals_5",
        ],
        "recent_form_10": [
            "home_wins_last_10", "home_win_rate_last_10",
            "home_goals_scored_last_10", "home_goals_conceded_last_10",
            "home_goal_diff_last_10", "home_clean_sheets_last_10",
            "away_wins_last_10", "away_win_rate_last_10",
            "away_goals_scored_last_10", "away_goals_conceded_last_10",
            "away_goal_diff_last_10", "away_clean_sheets_last_10",
            "form_diff_win_rate_10", "form_diff_goals_10",
        ],
        "squad_info": [
            "home_avg_age", "away_avg_age",
            "home_top5_players", "away_top5_players",
        ],
        "wc_history": [
            "home_wc_appearances", "away_wc_appearances",
            "home_wc_best_code", "away_wc_best_code",
            "home_wc_total_matches", "away_wc_total_matches",
            "home_wc_win_rate", "away_wc_win_rate",
        ],
    }
