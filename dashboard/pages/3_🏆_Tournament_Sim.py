"""
Tournament Simulator Page — Monte Carlo World Cup Simulation
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Tournament Sim | WC 2026", page_icon="🏆", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .gradient-divider { height: 2px; background: linear-gradient(90deg, transparent, #00ff87, #60efff, transparent); margin: 1.5rem 0; }
    .sim-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3561; border-radius: 16px; padding: 1.5rem; text-align: center;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🏆 Tournament Simulator</div>', unsafe_allow_html=True)
st.markdown("*Run Monte Carlo simulations to predict the World Cup winner*")
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Simulation controls
col1, col2 = st.columns([2, 1])

with col1:
    n_sims = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100,
                       help="More simulations = more accurate probabilities but slower computation")

with col2:
    st.markdown("""
    <div class="sim-card">
        <p style="color: #8892b0;">Estimated Time</p>
        <p style="font-size: 1.5rem; font-weight: 700; color: #00ff87;">~{} sec</p>
    </div>
    """.format(max(1, n_sims // 100)), unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

if st.button("🎲 Run Tournament Simulation", use_container_width=True, type="primary"):
    
    progress_bar = st.progress(0, text="Loading data...")
    
    # Load everything
    from src import data_loader, features
    
    matches_df = data_loader.load_matches()
    rankings_df = data_loader.load_rankings()
    elo_df = data_loader.load_elo_ratings()
    metadata_df = data_loader.load_team_metadata()
    groups = data_loader.load_wc_groups()
    
    progress_bar.progress(10, text="Loading model...")
    
    # Try to load trained model
    try:
        from src.model import load_model
        model, model_meta = load_model("xgb_match_predictor")
        model_loaded = True
    except:
        model = None
        model_loaded = False
    
    progress_bar.progress(20, text="Running simulation...")
    
    if model_loaded:
        # Use actual ML model with Monte Carlo simulation
        from src.simulator import run_full_tournament
        
        results = run_full_tournament(
            model, features.build_match_features, matches_df, groups,
            rankings_df, elo_df, metadata_df,
            n_simulations=n_sims, verbose=False
        )
        source = "XGBoost ML Model + Monte Carlo"
    else:
        # Elo-based simulation
        st.warning("⚠️ ML model not trained. Using Elo-based simulation. Run `python scripts/train_model.py` for better results.")
        
        # Simplified Elo-based tournament simulation
        results = {}
        total_elo_pow = sum(row["elo_rating"] ** 2 for _, row in metadata_df.iterrows())
        
        for _, row in metadata_df.iterrows():
            elo_share = (row["elo_rating"] ** 2) / total_elo_pow
            
            # Add randomness through simulation
            wins = 0
            finals = 0
            semis = 0
            for _ in range(n_sims):
                # Simplified: probability of winning decreases each round
                p = elo_share * 48  # Base probability scaled
                if np.random.random() < min(0.95, p * 1.5):  # Make semis
                    semis += 1
                    if np.random.random() < min(0.9, p * 1.2):  # Make final
                        finals += 1
                        if np.random.random() < min(0.85, p):  # Win
                            wins += 1
            
            results[row["team"]] = {
                "win_probability": round(wins / n_sims * 100, 2),
                "final_probability": round(finals / n_sims * 100, 2),
                "semifinal_probability": round(semis / n_sims * 100, 2),
                "wins": wins,
                "finals": finals,
                "semifinals": semis,
            }
        
        results = dict(sorted(results.items(), key=lambda x: x[1]["win_probability"], reverse=True))
        source = "Elo-Based Simulation"
    
    progress_bar.progress(100, text="Complete! ✅")
    
    # ===== RESULTS DISPLAY =====
    st.markdown(f"### 📊 Results ({n_sims:,} simulations)")
    st.caption(f"Source: {source}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["win_probability"], reverse=True)
    
    # Winner announcement
    winner = sorted_results[0]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #00ff87, #60efff); color: #000; 
                text-align: center; padding: 1.5rem; border-radius: 16px; margin: 1rem 0;">
        <div style="font-size: 1rem; font-weight: 500;">MOST LIKELY WINNER</div>
        <div style="font-size: 2.5rem; font-weight: 800;">🏆 {winner[0]}</div>
        <div style="font-size: 1.2rem; font-weight: 600;">{winner[1]['win_probability']:.1f}% Win Probability</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Podium
    col1, col2, col3 = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, col in enumerate([col1, col2, col3]):
        if i < len(sorted_results):
            team, stats = sorted_results[i]
            with col:
                st.markdown(f"""
                <div class="sim-card">
                    <div style="font-size: 2rem;">{medals[i]}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #e2e8f0;">{team}</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #00ff87;">{stats['win_probability']:.1f}%</div>
                    <div style="color: #8892b0; font-size: 0.85rem;">
                        Final: {stats['final_probability']:.1f}% | Semi: {stats['semifinal_probability']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== TOP 20 CHART =====
    st.markdown("### 📈 Win Probability Distribution (Top 20)")
    
    top_20 = sorted_results[:20]
    
    fig = go.Figure()
    
    teams = [t[0] for t in reversed(top_20)]
    win_probs = [t[1]["win_probability"] for t in reversed(top_20)]
    final_probs = [t[1]["final_probability"] for t in reversed(top_20)]
    semi_probs = [t[1]["semifinal_probability"] for t in reversed(top_20)]
    
    fig.add_trace(go.Bar(
        y=teams, x=semi_probs, name="Semi-Final",
        orientation='h', marker_color='rgba(96, 239, 255, 0.3)',
        hovertemplate="%{y}: %{x:.1f}%<extra>Semi-Final</extra>"
    ))
    fig.add_trace(go.Bar(
        y=teams, x=final_probs, name="Final",
        orientation='h', marker_color='rgba(255, 215, 0, 0.5)',
        hovertemplate="%{y}: %{x:.1f}%<extra>Final</extra>"
    ))
    fig.add_trace(go.Bar(
        y=teams, x=win_probs, name="Win",
        orientation='h', marker_color='#00ff87',
        hovertemplate="%{y}: %{x:.1f}%<extra>Win</extra>"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        barmode='overlay',
        xaxis=dict(title="Probability (%)", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        margin=dict(l=0, r=50, t=30, b=50),
        font=dict(family='Inter', size=12),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== FULL RESULTS TABLE =====
    with st.expander("📋 Full Results Table (All 48 Teams)"):
        results_df = pd.DataFrame([
            {
                "Rank": i+1,
                "Team": team,
                "Win %": f"{stats['win_probability']:.2f}",
                "Final %": f"{stats['final_probability']:.2f}",
                "Semi %": f"{stats['semifinal_probability']:.2f}",
                "Tournament Wins": stats["wins"],
                "Finals Reached": stats["finals"],
                "Semis Reached": stats["semifinals"],
            }
            for i, (team, stats) in enumerate(sorted_results)
        ])
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Confederation analysis
    with st.expander("🌍 Confederation Analysis"):
        conf_data = {}
        for team, stats in results.items():
            row = metadata_df[metadata_df["team"] == team]
            if len(row) > 0:
                conf = row.iloc[0]["confederation"]
                if conf not in conf_data:
                    conf_data[conf] = {"total_win_prob": 0, "teams": 0, "best_team": "", "best_prob": 0}
                conf_data[conf]["total_win_prob"] += stats["win_probability"]
                conf_data[conf]["teams"] += 1
                if stats["win_probability"] > conf_data[conf]["best_prob"]:
                    conf_data[conf]["best_team"] = team
                    conf_data[conf]["best_prob"] = stats["win_probability"]
        
        conf_df = pd.DataFrame([
            {"Confederation": conf, "Total Win %": f"{data['total_win_prob']:.1f}",
             "Teams": data["teams"], "Best Team": data["best_team"],
             "Best Team Win %": f"{data['best_prob']:.2f}"}
            for conf, data in sorted(conf_data.items(), key=lambda x: x[1]["total_win_prob"], reverse=True)
        ])
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    ### 🎲 How It Works
    
    The tournament simulator uses **Monte Carlo simulation** to predict World Cup outcomes:
    
    1. **Group Stage**: Each group's matches are simulated using the ML model's match predictions
    2. **Advancement**: Top 2 from each group + 8 best third-place teams advance (32 total)
    3. **Knockout Rounds**: R32 → R16 → QF → SF → Final, each match predicted by the model
    4. **Repeat N times**: The entire tournament is simulated thousands of times
    5. **Aggregate**: Count how many times each team wins → Win Probability
    
    > 💡 **Tip**: Start with 1,000 simulations for a quick estimate, then increase to 10,000 for more stable probabilities.
    """)
