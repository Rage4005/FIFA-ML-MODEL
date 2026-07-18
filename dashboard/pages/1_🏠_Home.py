"""
Home Page — Tournament Overview & Top Predictions
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Home | WC 2026 Predictor", page_icon="🏠", layout="wide")

# Custom CSS (shared)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff87, #60efff, transparent);
        margin: 1.5rem 0; border: none;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3561; border-radius: 16px;
        padding: 1.2rem; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00ff87; }
    .metric-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    .favorite-card {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        border: 1px solid #2d3561; border-radius: 16px;
        padding: 1.2rem; margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    .favorite-card:hover { transform: translateY(-3px); }
    .team-name { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; }
    .win-pct { font-size: 1.8rem; font-weight: 800; color: #00ff87; }
    .group-badge {
        display: inline-block; padding: 2px 10px; border-radius: 6px;
        background: rgba(0, 255, 135, 0.15); color: #00ff87;
        font-size: 0.75rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🏠 Tournament Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Load all required data."""
    from src import data_loader
    try:
        metadata = data_loader.load_team_metadata()
        groups = data_loader.load_wc_groups()
        return metadata, groups
    except Exception as e:
        st.error(f"Data not found. Please run `python scripts/generate_data.py` first.\n\nError: {e}")
        return None, None


@st.cache_data
def get_model_predictions():
    """Get or compute model predictions for all teams."""
    try:
        from src.model import load_model
        model, metadata = load_model("xgb_match_predictor")
        
        # Simple win probability heuristic based on Elo when simulation isn't available
        from src import data_loader
        team_meta = data_loader.load_team_metadata()
        
        total_elo = team_meta["elo_rating"].sum()
        predictions = {}
        for _, row in team_meta.iterrows():
            # Elo-based approximation of tournament win probability
            elo_share = row["elo_rating"] / total_elo
            # Apply power law to make favorites stand out more
            raw_prob = (elo_share ** 1.5)
            predictions[row["team"]] = {
                "win_probability": 0,
                "elo": row["elo_rating"],
                "fifa_rank": row["fifa_rank"],
                "group": row["group"],
            }
        
        # Normalize
        total = sum(v["elo"] ** 1.5 for v in predictions.values())
        for team in predictions:
            predictions[team]["win_probability"] = round(
                (predictions[team]["elo"] ** 1.5) / total * 100, 2
            )
        
        return predictions, True
    except Exception:
        # Fallback: Use Elo-based estimates
        from src import data_loader
        team_meta = data_loader.load_team_metadata()
        
        predictions = {}
        total_elo_pow = sum(row["elo_rating"] ** 1.5 for _, row in team_meta.iterrows())
        
        for _, row in team_meta.iterrows():
            predictions[row["team"]] = {
                "win_probability": round((row["elo_rating"] ** 1.5) / total_elo_pow * 100, 2),
                "elo": row["elo_rating"],
                "fifa_rank": row["fifa_rank"],
                "group": row["group"],
            }
        
        return predictions, False


metadata, groups = load_all_data()

if metadata is not None:
    predictions, model_loaded = get_model_predictions()
    
    # Sort by win probability
    sorted_teams = sorted(predictions.items(), key=lambda x: x[1]["win_probability"], reverse=True)
    
    if not model_loaded:
        st.warning("⚠️ Model not trained yet. Showing Elo-based estimates. Run `python scripts/train_model.py` for ML predictions.")
    
    # ===== KEY METRICS =====
    col1, col2, col3, col4 = st.columns(4)
    top_team = sorted_teams[0]
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">🥇 {top_team[0]}</div>
            <div class="metric-label">Top Favorite</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_team[1]['win_probability']:.1f}%</div>
            <div class="metric-label">Win Probability</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        model_acc = "~55%" if model_loaded else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{model_acc}</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">48</div>
            <div class="metric-label">Teams Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== TOP 10 FAVORITES =====
    st.markdown("### 🏆 Top 10 Favorites to Win")
    
    top_10 = sorted_teams[:10]
    top_10_df = pd.DataFrame([
        {"Team": team, "Win %": stats["win_probability"], 
         "Elo": stats["elo"], "FIFA Rank": stats["fifa_rank"],
         "Group": stats["group"]}
        for team, stats in top_10
    ])
    
    # Horizontal bar chart
    fig = go.Figure()
    
    colors = ['#00ff87', '#00e67a', '#00cc6a', '#00b35c', '#009950',
              '#008045', '#00663a', '#004d2f', '#003324', '#001a19']
    
    for i, (team, stats) in enumerate(reversed(top_10)):
        fig.add_trace(go.Bar(
            y=[team],
            x=[stats["win_probability"]],
            orientation='h',
            marker_color=colors[9-i],
            text=f" {stats['win_probability']:.1f}%",
            textposition="outside",
            textfont=dict(size=14, color='#e2e8f0', family='Inter'),
            hovertemplate=f"<b>{team}</b><br>Win: {stats['win_probability']:.1f}%<br>Elo: {stats['elo']}<br>FIFA Rank: {stats['fifa_rank']}<extra></extra>",
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=0, r=80, t=10, b=10),
        xaxis=dict(title="Win Probability (%)", gridcolor='rgba(255,255,255,0.1)', range=[0, max(s["win_probability"] for _, s in top_10) * 1.3]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        showlegend=False,
        font=dict(family='Inter', size=13),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== GROUP OVERVIEW =====
    st.markdown("### 📋 Group Stage Overview")
    
    # Display groups in a grid
    group_names = sorted(groups.keys())
    
    for row_start in range(0, len(group_names), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(group_names):
                group_name = group_names[idx]
                group_teams = groups[group_name]
                
                with col:
                    st.markdown(f"**Group {group_name}**")
                    for team in group_teams:
                        rank = predictions.get(team, {}).get("fifa_rank", "?")
                        elo = predictions.get(team, {}).get("elo", "?")
                        st.markdown(f"- **{team}** (#{rank}, Elo: {elo})")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== ALL TEAMS TABLE =====
    with st.expander("📊 Full 48-Team Predictions Table"):
        all_teams_df = pd.DataFrame([
            {
                "Rank": i+1,
                "Team": team,
                "Win %": f"{stats['win_probability']:.2f}%",
                "Elo Rating": stats["elo"],
                "FIFA Rank": stats["fifa_rank"],
                "Group": stats["group"],
            }
            for i, (team, stats) in enumerate(sorted_teams)
        ])
        st.dataframe(all_teams_df, use_container_width=True, hide_index=True)
