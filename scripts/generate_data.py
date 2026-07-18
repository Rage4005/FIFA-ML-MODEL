"""
FIFA World Cup 2026 — Comprehensive Dataset Generator

Generates synthetic-but-realistic training data based on real-world football statistics.
This module creates all the data files needed for the ML pipeline:
- International match results (historical patterns)
- FIFA rankings
- Elo ratings  
- Team metadata (squad info, market values)
- World Cup 2026 groups and schedule

In production, replace this with actual Kaggle dataset downloads.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# ============================================================================
# REAL-WORLD REFERENCE DATA
# ============================================================================

# All 48 teams in the 2026 World Cup with realistic attributes
WORLD_CUP_2026_TEAMS = {
    # Group A
    "Canada": {"confederation": "CONCACAF", "group": "A", "fifa_rank": 43, "elo": 1677, "market_value_b": 0.28, "avg_age": 27.1, "top5_league_players": 11, "wc_appearances": 3, "wc_best": "Group Stage", "strength_tier": 3},
    "Argentina": {"confederation": "CONMEBOL", "group": "A", "fifa_rank": 1, "elo": 2143, "market_value_b": 0.93, "avg_age": 28.5, "top5_league_players": 22, "wc_appearances": 18, "wc_best": "Winner", "strength_tier": 1},
    "Morocco": {"confederation": "CAF", "group": "A", "fifa_rank": 13, "elo": 1901, "market_value_b": 0.48, "avg_age": 27.3, "top5_league_players": 18, "wc_appearances": 7, "wc_best": "Semi-Finals", "strength_tier": 2},
    "Uzbekistan": {"confederation": "AFC", "group": "A", "fifa_rank": 62, "elo": 1581, "market_value_b": 0.05, "avg_age": 26.8, "top5_league_players": 2, "wc_appearances": 1, "wc_best": "Group Stage", "strength_tier": 4},

    # Group B
    "Mexico": {"confederation": "CONCACAF", "group": "B", "fifa_rank": 14, "elo": 1868, "market_value_b": 0.27, "avg_age": 27.6, "top5_league_players": 8, "wc_appearances": 17, "wc_best": "Quarter-Finals", "strength_tier": 2},
    "Ecuador": {"confederation": "CONMEBOL", "group": "B", "fifa_rank": 30, "elo": 1772, "market_value_b": 0.22, "avg_age": 26.2, "top5_league_players": 12, "wc_appearances": 4, "wc_best": "Round of 16", "strength_tier": 3},
    "Senegal": {"confederation": "CAF", "group": "B", "fifa_rank": 20, "elo": 1826, "market_value_b": 0.35, "avg_age": 26.9, "top5_league_players": 19, "wc_appearances": 4, "wc_best": "Quarter-Finals", "strength_tier": 2},
    "Bolivia": {"confederation": "CONMEBOL", "group": "B", "fifa_rank": 76, "elo": 1487, "market_value_b": 0.02, "avg_age": 27.8, "top5_league_players": 1, "wc_appearances": 4, "wc_best": "Group Stage", "strength_tier": 4},

    # Group C
    "United States": {"confederation": "CONCACAF", "group": "C", "fifa_rank": 11, "elo": 1920, "market_value_b": 0.51, "avg_age": 26.4, "top5_league_players": 14, "wc_appearances": 12, "wc_best": "Third Place", "strength_tier": 2},
    "Colombia": {"confederation": "CONMEBOL", "group": "C", "fifa_rank": 12, "elo": 1904, "market_value_b": 0.42, "avg_age": 27.7, "top5_league_players": 15, "wc_appearances": 7, "wc_best": "Quarter-Finals", "strength_tier": 2},
    "Egypt": {"confederation": "CAF", "group": "C", "fifa_rank": 33, "elo": 1737, "market_value_b": 0.18, "avg_age": 27.5, "top5_league_players": 7, "wc_appearances": 4, "wc_best": "Group Stage", "strength_tier": 3},
    "New Zealand": {"confederation": "OFC", "group": "C", "fifa_rank": 93, "elo": 1482, "market_value_b": 0.01, "avg_age": 27.0, "top5_league_players": 2, "wc_appearances": 3, "wc_best": "Group Stage", "strength_tier": 4},

    # Group D
    "Brazil": {"confederation": "CONMEBOL", "group": "D", "fifa_rank": 5, "elo": 2074, "market_value_b": 1.18, "avg_age": 27.0, "top5_league_players": 21, "wc_appearances": 22, "wc_best": "Winner", "strength_tier": 1},
    "Italy": {"confederation": "UEFA", "group": "D", "fifa_rank": 9, "elo": 1944, "market_value_b": 0.68, "avg_age": 27.8, "top5_league_players": 23, "wc_appearances": 18, "wc_best": "Winner", "strength_tier": 1},
    "Paraguay": {"confederation": "CONMEBOL", "group": "D", "fifa_rank": 45, "elo": 1645, "market_value_b": 0.09, "avg_age": 26.5, "top5_league_players": 6, "wc_appearances": 9, "wc_best": "Quarter-Finals", "strength_tier": 3},
    "Cameroon": {"confederation": "CAF", "group": "D", "fifa_rank": 48, "elo": 1621, "market_value_b": 0.14, "avg_age": 27.2, "top5_league_players": 12, "wc_appearances": 8, "wc_best": "Quarter-Finals", "strength_tier": 3},

    # Group E
    "France": {"confederation": "UEFA", "group": "E", "fifa_rank": 2, "elo": 2112, "market_value_b": 1.32, "avg_age": 27.3, "top5_league_players": 23, "wc_appearances": 16, "wc_best": "Winner", "strength_tier": 1},
    "Australia": {"confederation": "AFC", "group": "E", "fifa_rank": 24, "elo": 1793, "market_value_b": 0.10, "avg_age": 27.4, "top5_league_players": 9, "wc_appearances": 7, "wc_best": "Round of 16", "strength_tier": 3},
    "Indonesia": {"confederation": "AFC", "group": "E", "fifa_rank": 87, "elo": 1497, "market_value_b": 0.02, "avg_age": 26.1, "top5_league_players": 1, "wc_appearances": 1, "wc_best": "Group Stage", "strength_tier": 4},
    "Honduras": {"confederation": "CONCACAF", "group": "E", "fifa_rank": 68, "elo": 1532, "market_value_b": 0.03, "avg_age": 27.5, "top5_league_players": 3, "wc_appearances": 4, "wc_best": "Group Stage", "strength_tier": 4},

    # Group F
    "Spain": {"confederation": "UEFA", "group": "F", "fifa_rank": 3, "elo": 2078, "market_value_b": 1.36, "avg_age": 26.9, "top5_league_players": 23, "wc_appearances": 16, "wc_best": "Winner", "strength_tier": 1},
    "Nigeria": {"confederation": "CAF", "group": "F", "fifa_rank": 28, "elo": 1779, "market_value_b": 0.25, "avg_age": 26.7, "top5_league_players": 15, "wc_appearances": 7, "wc_best": "Round of 16", "strength_tier": 2},
    "Peru": {"confederation": "CONMEBOL", "group": "F", "fifa_rank": 32, "elo": 1742, "market_value_b": 0.10, "avg_age": 28.1, "top5_league_players": 7, "wc_appearances": 6, "wc_best": "Quarter-Finals", "strength_tier": 3},
    "Albania": {"confederation": "UEFA", "group": "F", "fifa_rank": 56, "elo": 1600, "market_value_b": 0.10, "avg_age": 27.0, "top5_league_players": 10, "wc_appearances": 2, "wc_best": "Group Stage", "strength_tier": 3},

    # Group G
    "England": {"confederation": "UEFA", "group": "G", "fifa_rank": 4, "elo": 2064, "market_value_b": 1.52, "avg_age": 26.8, "top5_league_players": 23, "wc_appearances": 16, "wc_best": "Winner", "strength_tier": 1},
    "Serbia": {"confederation": "UEFA", "group": "G", "fifa_rank": 34, "elo": 1746, "market_value_b": 0.26, "avg_age": 27.6, "top5_league_players": 16, "wc_appearances": 13, "wc_best": "Semi-Finals", "strength_tier": 2},
    "Qatar": {"confederation": "AFC", "group": "G", "fifa_rank": 39, "elo": 1692, "market_value_b": 0.04, "avg_age": 28.2, "top5_league_players": 1, "wc_appearances": 2, "wc_best": "Group Stage", "strength_tier": 3},
    "Chile": {"confederation": "CONMEBOL", "group": "G", "fifa_rank": 36, "elo": 1731, "market_value_b": 0.12, "avg_age": 28.8, "top5_league_players": 8, "wc_appearances": 10, "wc_best": "Third Place", "strength_tier": 3},

    # Group H
    "Germany": {"confederation": "UEFA", "group": "H", "fifa_rank": 7, "elo": 1988, "market_value_b": 0.95, "avg_age": 27.1, "top5_league_players": 23, "wc_appearances": 20, "wc_best": "Winner", "strength_tier": 1},
    "Japan": {"confederation": "AFC", "group": "H", "fifa_rank": 15, "elo": 1862, "market_value_b": 0.22, "avg_age": 27.3, "top5_league_players": 18, "wc_appearances": 8, "wc_best": "Round of 16", "strength_tier": 2},
    "Costa Rica": {"confederation": "CONCACAF", "group": "H", "fifa_rank": 53, "elo": 1609, "market_value_b": 0.04, "avg_age": 27.9, "top5_league_players": 4, "wc_appearances": 6, "wc_best": "Quarter-Finals", "strength_tier": 3},
    "Tunisia": {"confederation": "CAF", "group": "H", "fifa_rank": 38, "elo": 1698, "market_value_b": 0.06, "avg_age": 27.5, "top5_league_players": 9, "wc_appearances": 7, "wc_best": "Group Stage", "strength_tier": 3},

    # Group I
    "Portugal": {"confederation": "UEFA", "group": "I", "fifa_rank": 6, "elo": 2040, "market_value_b": 1.05, "avg_age": 27.8, "top5_league_players": 20, "wc_appearances": 8, "wc_best": "Third Place", "strength_tier": 1},
    "Uruguay": {"confederation": "CONMEBOL", "group": "I", "fifa_rank": 10, "elo": 1936, "market_value_b": 0.36, "avg_age": 27.6, "top5_league_players": 16, "wc_appearances": 14, "wc_best": "Winner", "strength_tier": 1},
    "Panama": {"confederation": "CONCACAF", "group": "I", "fifa_rank": 46, "elo": 1640, "market_value_b": 0.03, "avg_age": 27.8, "top5_league_players": 3, "wc_appearances": 3, "wc_best": "Group Stage", "strength_tier": 3},
    "South Korea": {"confederation": "AFC", "group": "I", "fifa_rank": 22, "elo": 1812, "market_value_b": 0.18, "avg_age": 27.1, "top5_league_players": 10, "wc_appearances": 11, "wc_best": "Semi-Finals", "strength_tier": 2},

    # Group J
    "Netherlands": {"confederation": "UEFA", "group": "J", "fifa_rank": 8, "elo": 1975, "market_value_b": 0.82, "avg_age": 27.0, "top5_league_players": 21, "wc_appearances": 11, "wc_best": "Runner-Up", "strength_tier": 1},
    "Iran": {"confederation": "AFC", "group": "J", "fifa_rank": 21, "elo": 1819, "market_value_b": 0.08, "avg_age": 28.0, "top5_league_players": 6, "wc_appearances": 7, "wc_best": "Group Stage", "strength_tier": 3},
    "Wales": {"confederation": "UEFA", "group": "J", "fifa_rank": 41, "elo": 1680, "market_value_b": 0.12, "avg_age": 27.4, "top5_league_players": 12, "wc_appearances": 3, "wc_best": "Quarter-Finals", "strength_tier": 3},
    "Tanzania": {"confederation": "CAF", "group": "J", "fifa_rank": 71, "elo": 1518, "market_value_b": 0.01, "avg_age": 26.3, "top5_league_players": 2, "wc_appearances": 1, "wc_best": "Group Stage", "strength_tier": 4},

    # Group K
    "Belgium": {"confederation": "UEFA", "group": "K", "fifa_rank": 16, "elo": 1858, "market_value_b": 0.52, "avg_age": 27.9, "top5_league_players": 20, "wc_appearances": 14, "wc_best": "Third Place", "strength_tier": 2},
    "Saudi Arabia": {"confederation": "AFC", "group": "K", "fifa_rank": 55, "elo": 1607, "market_value_b": 0.08, "avg_age": 27.7, "top5_league_players": 2, "wc_appearances": 7, "wc_best": "Round of 16", "strength_tier": 3},
    "Jamaica": {"confederation": "CONCACAF", "group": "K", "fifa_rank": 57, "elo": 1594, "market_value_b": 0.06, "avg_age": 26.8, "top5_league_players": 8, "wc_appearances": 2, "wc_best": "Group Stage", "strength_tier": 3},
    "Denmark": {"confederation": "UEFA", "group": "K", "fifa_rank": 19, "elo": 1838, "market_value_b": 0.42, "avg_age": 27.2, "top5_league_players": 19, "wc_appearances": 6, "wc_best": "Quarter-Finals", "strength_tier": 2},

    # Group L
    "Croatia": {"confederation": "UEFA", "group": "L", "fifa_rank": 17, "elo": 1853, "market_value_b": 0.38, "avg_age": 28.3, "top5_league_players": 18, "wc_appearances": 7, "wc_best": "Runner-Up", "strength_tier": 2},
    "Turkey": {"confederation": "UEFA", "group": "L", "fifa_rank": 25, "elo": 1786, "market_value_b": 0.35, "avg_age": 26.5, "top5_league_players": 12, "wc_appearances": 3, "wc_best": "Third Place", "strength_tier": 2},
    "Scotland": {"confederation": "UEFA", "group": "L", "fifa_rank": 42, "elo": 1685, "market_value_b": 0.15, "avg_age": 27.6, "top5_league_players": 14, "wc_appearances": 9, "wc_best": "Group Stage", "strength_tier": 3},
    "Ghana": {"confederation": "CAF", "group": "L", "fifa_rank": 50, "elo": 1612, "market_value_b": 0.11, "avg_age": 26.4, "top5_league_players": 13, "wc_appearances": 5, "wc_best": "Quarter-Finals", "strength_tier": 3},
}

# WC Best finish numerical encoding
WC_BEST_ENCODING = {
    "Winner": 7,
    "Runner-Up": 6,
    "Third Place": 5,
    "Semi-Finals": 5,
    "Quarter-Finals": 4,
    "Round of 16": 3,
    "Group Stage": 1,
    "Never": 0,
}


def generate_historical_matches(n_matches=8000, seed=42):
    """
    Generate realistic historical international match results.
    Models real-world patterns:
    - Home advantage (~46% home win, ~26% draw, ~28% away win)
    - Stronger teams (lower Elo) win more often
    - Goals follow ~Poisson distribution (avg ~1.3 per team per match)
    - Tournament type affects competitiveness
    """
    np.random.seed(seed)
    
    teams = list(WORLD_CUP_2026_TEAMS.keys())
    all_international_teams = teams + [
        "Russia", "Sweden", "Switzerland", "Austria", "Czech Republic", 
        "Norway", "Ukraine", "Romania", "Poland", "Greece",
        "Hungary", "Ireland", "Iceland", "Slovakia", "Slovenia",
        "North Macedonia", "Georgia", "Finland", "Luxembourg", "Cyprus",
        "China", "India", "Thailand", "Vietnam", "Iraq",
        "Oman", "Bahrain", "Jordan", "Syria", "Lebanon",
        "Mali", "Ivory Coast", "Algeria", "DR Congo", "Burkina Faso",
        "South Africa", "Zimbabwe", "Zambia", "Kenya", "Uganda",
        "Venezuela", "Cuba", "Dominican Republic", "Guatemala", "El Salvador",
        "Trinidad and Tobago", "Haiti", "Nicaragua", "Bermuda", "Curacao"
    ]
    
    # Assign Elo ratings to non-WC teams
    team_elos = {}
    for team in teams:
        team_elos[team] = WORLD_CUP_2026_TEAMS[team]["elo"]
    for team in all_international_teams:
        if team not in team_elos:
            team_elos[team] = np.random.randint(1200, 1700)
    
    tournaments = [
        ("FIFA World Cup", 0.15),
        ("FIFA World Cup qualification", 0.30),
        ("Continental Championship", 0.10),
        ("Continental Championship qualification", 0.15),
        ("Friendly", 0.25),
        ("Nations League", 0.05),
    ]
    tournament_names = [t[0] for t in tournaments]
    tournament_probs = [t[1] for t in tournaments]
    
    matches = []
    start_date = datetime(2010, 1, 1)
    
    for i in range(n_matches):
        # Pick two different teams
        home_team = np.random.choice(all_international_teams)
        away_candidates = [t for t in all_international_teams if t != home_team]
        
        # Weight towards teams from same confederation or WC teams
        away_team = np.random.choice(away_candidates)
        
        # Generate date
        days_offset = np.random.randint(0, (datetime(2026, 6, 1) - start_date).days)
        match_date = start_date + timedelta(days=days_offset)
        
        # Tournament type
        tournament = np.random.choice(tournament_names, p=tournament_probs)
        
        # Calculate expected goals based on Elo difference
        home_elo = team_elos.get(home_team, 1500)
        away_elo = team_elos.get(away_team, 1500)
        
        elo_diff = home_elo - away_elo
        
        # Home advantage + Elo-based expected goals
        home_advantage = 0.25 if tournament != "Friendly" else 0.15
        
        home_xg = max(0.3, 1.3 + (elo_diff / 800) + home_advantage)
        away_xg = max(0.3, 1.3 - (elo_diff / 800))
        
        # Generate goals from Poisson distribution
        home_score = np.random.poisson(home_xg)
        away_score = np.random.poisson(away_xg)
        
        # Cap at reasonable scores
        home_score = min(home_score, 8)
        away_score = min(away_score, 8)
        
        # Neutral venue for some tournaments
        neutral = tournament in ["FIFA World Cup", "Continental Championship"]
        
        city = "Neutral Venue" if neutral else f"{home_team} City"
        country = "Neutral" if neutral else home_team
        
        matches.append({
            "date": match_date.strftime("%Y-%m-%d"),
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "city": city,
            "country": country,
            "neutral": neutral,
        })
    
    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def generate_fifa_rankings(teams_dict, start_year=2010, end_year=2026):
    """Generate historical FIFA rankings data with realistic evolution."""
    np.random.seed(123)
    
    rankings = []
    teams = list(teams_dict.keys())
    
    for year in range(start_year, end_year + 1):
        for month in [1, 4, 7, 10]:
            date = datetime(year, month, 1)
            
            team_points = {}
            for team in teams:
                base = teams_dict[team]["elo"]
                # Add temporal noise (teams improve/decline over years)
                trend = (year - 2018) * np.random.uniform(-5, 5)
                noise = np.random.normal(0, 30)
                points = max(800, base + trend + noise)
                team_points[team] = points
            
            # Sort by points to get rank
            sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (team, points) in enumerate(sorted_teams, 1):
                rankings.append({
                    "rank": rank,
                    "country_full": team,
                    "total_points": round(points, 2),
                    "rank_date": date.strftime("%Y-%m-%d"),
                    "confederation": teams_dict[team]["confederation"],
                })
    
    return pd.DataFrame(rankings)


def generate_elo_ratings(teams_dict, start_year=2010, end_year=2026):
    """Generate historical Elo ratings data."""
    np.random.seed(456)
    
    elo_data = []
    teams = list(teams_dict.keys())
    
    for year in range(start_year, end_year + 1):
        for team in teams:
            base_elo = teams_dict[team]["elo"]
            # Simulate Elo evolution
            yearly_shift = (year - 2020) * np.random.uniform(-8, 8)
            noise = np.random.normal(0, 20)
            elo = max(1200, int(base_elo + yearly_shift + noise))
            
            elo_data.append({
                "team": team,
                "year": year,
                "elo_rating": elo,
                "confederation": teams_dict[team]["confederation"],
            })
    
    return pd.DataFrame(elo_data)


def generate_team_metadata():
    """Generate team metadata including squad info, market values, etc."""
    metadata = []
    for team, info in WORLD_CUP_2026_TEAMS.items():
        metadata.append({
            "team": team,
            "confederation": info["confederation"],
            "group": info["group"],
            "fifa_rank": info["fifa_rank"],
            "elo_rating": info["elo"],
            "market_value_billion_eur": info["market_value_b"],
            "avg_squad_age": info["avg_age"],
            "top5_league_players": info["top5_league_players"],
            "wc_appearances": info["wc_appearances"],
            "wc_best_finish": info["wc_best"],
            "wc_best_finish_code": WC_BEST_ENCODING[info["wc_best"]],
            "strength_tier": info["strength_tier"],
        })
    return pd.DataFrame(metadata)


def generate_all_data(output_dir="data"):
    """Generate all datasets and save to disk."""
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("📊 Generating historical match results...")
    matches = generate_historical_matches(n_matches=10000)
    matches.to_csv(os.path.join(raw_dir, "international_matches.csv"), index=False)
    print(f"   ✅ {len(matches)} matches generated")
    
    print("🏅 Generating FIFA rankings...")
    rankings = generate_fifa_rankings(WORLD_CUP_2026_TEAMS)
    rankings.to_csv(os.path.join(raw_dir, "fifa_rankings.csv"), index=False)
    print(f"   ✅ {len(rankings)} ranking records generated")
    
    print("📈 Generating Elo ratings...")
    elo = generate_elo_ratings(WORLD_CUP_2026_TEAMS)
    elo.to_csv(os.path.join(raw_dir, "elo_ratings.csv"), index=False)
    print(f"   ✅ {len(elo)} Elo records generated")
    
    print("👥 Generating team metadata...")
    metadata = generate_team_metadata()
    metadata.to_csv(os.path.join(raw_dir, "team_metadata.csv"), index=False)
    print(f"   ✅ {len(metadata)} teams")
    
    # Save WC 2026 groups as JSON
    groups = {}
    for team, info in WORLD_CUP_2026_TEAMS.items():
        group = info["group"]
        if group not in groups:
            groups[group] = []
        groups[group].append(team)
    
    with open(os.path.join(raw_dir, "wc2026_groups.json"), "w") as f:
        json.dump(groups, f, indent=2)
    print("   ✅ WC 2026 groups saved")
    
    # Save key player injuries (user-editable)
    injuries = {
        "Argentina": {"injured_key_players": 0, "details": []},
        "Brazil": {"injured_key_players": 1, "details": ["Neymar (knee)"]},
        "France": {"injured_key_players": 0, "details": []},
        "England": {"injured_key_players": 1, "details": ["Bukayo Saka (hamstring)"]},
        "Spain": {"injured_key_players": 0, "details": []},
        "Germany": {"injured_key_players": 0, "details": []},
        "Portugal": {"injured_key_players": 0, "details": []},
        "Netherlands": {"injured_key_players": 0, "details": []},
    }
    with open(os.path.join(raw_dir, "injuries.json"), "w") as f:
        json.dump(injuries, f, indent=2)
    print("   ✅ Injury data saved (user-editable)")
    
    print(f"\n🎉 All data saved to '{output_dir}/' directory!")
    return matches, rankings, elo, metadata


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate_all_data(output_dir)
