"""
Data Loader Module — FIFA World Cup 2026 Prediction

Handles loading, cleaning, and merging all datasets.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path


# Find project root (two levels up from src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_matches(filepath=None):
    """Load and clean international match results."""
    if filepath is None:
        filepath = RAW_DIR / "international_matches.csv"
    
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    
    # Add derived columns
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["draw"] = (df["home_score"] == df["away_score"]).astype(int)
    df["away_win"] = (df["home_score"] < df["away_score"]).astype(int)
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["year"] = df["date"].dt.year
    
    # Outcome label: 0=home_win, 1=draw, 2=away_win
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df["outcome"] = np.select(conditions, [0, 1, 2])
    
    return df


def load_rankings(filepath=None):
    """Load FIFA rankings data."""
    if filepath is None:
        filepath = RAW_DIR / "fifa_rankings.csv"
    
    df = pd.read_csv(filepath)
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    return df


def load_elo_ratings(filepath=None):
    """Load Elo ratings data."""
    if filepath is None:
        filepath = RAW_DIR / "elo_ratings.csv"
    
    return pd.read_csv(filepath)


def load_team_metadata(filepath=None):
    """Load team metadata."""
    if filepath is None:
        filepath = RAW_DIR / "team_metadata.csv"
    
    return pd.read_csv(filepath)


def load_wc_groups(filepath=None):
    """Load World Cup 2026 group assignments."""
    if filepath is None:
        filepath = RAW_DIR / "wc2026_groups.json"
    
    with open(filepath, "r") as f:
        return json.load(f)


def load_injuries(filepath=None):
    """Load injury data (user-editable)."""
    if filepath is None:
        filepath = RAW_DIR / "injuries.json"
    
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, "r") as f:
        return json.load(f)


def get_ranking_at_date(rankings_df, team, date):
    """Get a team's FIFA ranking closest to (but before) a given date."""
    team_rankings = rankings_df[
        (rankings_df["country_full"] == team) & 
        (rankings_df["rank_date"] <= date)
    ]
    
    if len(team_rankings) == 0:
        return {"rank": 100, "total_points": 1200.0}
    
    latest = team_rankings.sort_values("rank_date").iloc[-1]
    return {
        "rank": int(latest["rank"]),
        "total_points": float(latest["total_points"]),
    }


def get_elo_at_year(elo_df, team, year):
    """Get a team's Elo rating for a given year."""
    row = elo_df[(elo_df["team"] == team) & (elo_df["year"] == year)]
    
    if len(row) == 0:
        return 1500  # Default Elo
    
    return int(row.iloc[0]["elo_rating"])


def get_team_matches(matches_df, team, before_date=None, n_last=None):
    """Get all matches involving a specific team."""
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    
    if before_date is not None:
        mask = mask & (matches_df["date"] < before_date)
    
    team_matches = matches_df[mask].sort_values("date", ascending=False)
    
    if n_last is not None:
        team_matches = team_matches.head(n_last)
    
    return team_matches


def get_head_to_head(matches_df, team_a, team_b, n_last=10):
    """Get head-to-head record between two teams."""
    mask = (
        ((matches_df["home_team"] == team_a) & (matches_df["away_team"] == team_b)) |
        ((matches_df["home_team"] == team_b) & (matches_df["away_team"] == team_a))
    )
    
    h2h = matches_df[mask].sort_values("date", ascending=False).head(n_last)
    
    if len(h2h) == 0:
        return {"matches": 0, "team_a_wins": 0, "team_b_wins": 0, "draws": 0, "details": []}
    
    team_a_wins = 0
    team_b_wins = 0
    draws = 0
    details = []
    
    for _, match in h2h.iterrows():
        if match["home_team"] == team_a:
            if match["home_score"] > match["away_score"]:
                team_a_wins += 1
            elif match["home_score"] < match["away_score"]:
                team_b_wins += 1
            else:
                draws += 1
            details.append({
                "date": match["date"].strftime("%Y-%m-%d"),
                "score": f"{team_a} {match['home_score']}-{match['away_score']} {team_b}",
                "tournament": match["tournament"],
            })
        else:
            if match["away_score"] > match["home_score"]:
                team_a_wins += 1
            elif match["away_score"] < match["home_score"]:
                team_b_wins += 1
            else:
                draws += 1
            details.append({
                "date": match["date"].strftime("%Y-%m-%d"),
                "score": f"{team_b} {match['home_score']}-{match['away_score']} {team_a}",
                "tournament": match["tournament"],
            })
    
    return {
        "matches": len(h2h),
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "draws": draws,
        "details": details,
    }


def merge_match_with_rankings(matches_df, rankings_df):
    """Merge match data with FIFA rankings at the time of each match."""
    enriched = matches_df.copy()
    
    # Pre-compute ranking lookups for efficiency
    home_ranks = []
    home_points = []
    away_ranks = []
    away_points = []
    
    for _, match in enriched.iterrows():
        hr = get_ranking_at_date(rankings_df, match["home_team"], match["date"])
        ar = get_ranking_at_date(rankings_df, match["away_team"], match["date"])
        home_ranks.append(hr["rank"])
        home_points.append(hr["total_points"])
        away_ranks.append(ar["rank"])
        away_points.append(ar["total_points"])
    
    enriched["home_fifa_rank"] = home_ranks
    enriched["home_fifa_points"] = home_points
    enriched["away_fifa_rank"] = away_ranks
    enriched["away_fifa_points"] = away_points
    enriched["rank_diff"] = enriched["home_fifa_rank"] - enriched["away_fifa_rank"]
    enriched["points_diff"] = enriched["home_fifa_points"] - enriched["away_fifa_points"]
    
    return enriched


def get_all_wc_teams():
    """Return list of all 48 World Cup 2026 teams."""
    metadata = load_team_metadata()
    return sorted(metadata["team"].tolist())
