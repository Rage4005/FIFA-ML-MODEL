"""
Match Predictor Page — Head-to-Head Prediction with Side-by-Side Stats
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Match Predictor | WC 2026", page_icon="⚽", layout="wide")

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
    .vs-text { font-size: 3rem; font-weight: 800; color: #ffd700; text-align: center; }
    .team-header { font-size: 1.8rem; font-weight: 700; text-align: center; }
    .prob-display { text-align: center; padding: 1rem; }
    .prob-big { font-size: 3rem; font-weight: 800; }
    .prob-label { font-size: 0.9rem; color: #8892b0; text-transform: uppercase; }
    .winner-banner {
        background: linear-gradient(135deg, #00ff87, #60efff);
        color: #000; text-align: center; padding: 1rem; border-radius: 12px;
        font-size: 1.3rem; font-weight: 700; margin: 1rem 0;
    }
    .stat-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .stat-left { text-align: left; font-weight: 600; min-width: 80px; font-size: 1.05rem; }
    .stat-center { text-align: center; color: #8892b0; font-size: 0.85rem; flex: 1; }
    .stat-right { text-align: right; font-weight: 600; min-width: 80px; font-size: 1.05rem; }
    .better { color: #00ff87; }
    .worse { color: #ff4757; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">⚽ Match Predictor</div>', unsafe_allow_html=True)
st.markdown("*Select two teams and get AI-powered match predictions*")
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


@st.cache_data
def get_team_list():
    from src import data_loader
    try:
        return data_loader.get_all_wc_teams()
    except:
        return []


@st.cache_resource
def load_prediction_model():
    try:
        from src.model import load_model
        model, metadata = load_model("xgb_match_predictor")
        return model, metadata, True
    except:
        return None, None, False


teams = get_team_list()

if not teams:
    st.error("❌ No team data found. Run `python scripts/generate_data.py` first.")
    st.stop()

# Team selection
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    team_a = st.selectbox("🏠 Select Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0, key="team_a_select")

with col2:
    st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

with col3:
    default_b = teams.index("Spain") if "Spain" in teams else 1
    team_b = st.selectbox("✈️ Select Team B", teams, index=default_b, key="team_b_select")

if team_a == team_b:
    st.warning("Please select two different teams!")
    st.stop()

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Predict button
if st.button("🔮 Predict Match Outcome", use_container_width=True, type="primary"):
    
    with st.spinner("🧠 Running prediction model..."):
        from src import data_loader, features
        
        matches_df = data_loader.load_matches()
        rankings_df = data_loader.load_rankings()
        elo_df = data_loader.load_elo_ratings()
        metadata_df = data_loader.load_team_metadata()
        
        # Try ML model first, fall back to Elo-based
        model, model_meta, model_loaded = load_prediction_model()
        
        # Build features
        match_date = pd.Timestamp.now()
        feats = features.build_match_features(
            matches_df, team_a, team_b, match_date,
            rankings_df, elo_df, metadata_df
        )
        
        if model_loaded and model is not None:
            X = pd.DataFrame([feats]).fillna(0)
            
            # Align features
            if hasattr(model, 'get_booster'):
                model_features = model.get_booster().feature_names
                if model_features:
                    for col in model_features:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[model_features]
            
            proba = model.predict_proba(X)[0]
            prob_a = float(proba[0]) * 100
            prob_draw = float(proba[1]) * 100
            prob_b = float(proba[2]) * 100
            source = "XGBoost ML Model"
        else:
            # Elo-based fallback
            elo_a = feats.get("home_elo", 1500)
            elo_b = feats.get("away_elo", 1500)
            elo_diff = elo_a - elo_b
            
            # Bradley-Terry model approximation
            import numpy as np
            expected_a = 1 / (1 + 10 ** (-elo_diff / 400))
            prob_a = expected_a * 75  # Scale to leave room for draws
            prob_b = (1 - expected_a) * 75
            prob_draw = 100 - prob_a - prob_b
            source = "Elo-Based Estimate"
        
        # Determine winner
        if prob_a > prob_b and prob_a > prob_draw:
            predicted_winner = team_a
            confidence = prob_a
        elif prob_b > prob_a and prob_b > prob_draw:
            predicted_winner = team_b
            confidence = prob_b
        else:
            predicted_winner = "Draw"
            confidence = prob_draw
        
        # ===== RESULTS DISPLAY =====
        
        # Winner banner
        if predicted_winner == "Draw":
            st.markdown(f'<div class="winner-banner">🤝 Predicted Result: DRAW ({confidence:.1f}% probability)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="winner-banner">🏆 Predicted Winner: {predicted_winner} ({confidence:.1f}% confidence)</div>', unsafe_allow_html=True)
        
        st.caption(f"📊 Source: {source}")
        
        # Probability display
        col1, col2, col3 = st.columns(3)
        with col1:
            color_a = "#00ff87" if prob_a >= max(prob_b, prob_draw) else "#8892b0"
            st.markdown(f'<div class="prob-display"><div class="prob-big" style="color: {color_a};">{prob_a:.1f}%</div><div class="prob-label">{team_a} Win</div></div>', unsafe_allow_html=True)
        with col2:
            color_d = "#ffd700" if prob_draw >= max(prob_a, prob_b) else "#8892b0"
            st.markdown(f'<div class="prob-display"><div class="prob-big" style="color: {color_d};">{prob_draw:.1f}%</div><div class="prob-label">Draw</div></div>', unsafe_allow_html=True)
        with col3:
            color_b = "#60efff" if prob_b >= max(prob_a, prob_draw) else "#8892b0"
            st.markdown(f'<div class="prob-display"><div class="prob-big" style="color: {color_b};">{prob_b:.1f}%</div><div class="prob-label">{team_b} Win</div></div>', unsafe_allow_html=True)
        
        # Probability bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[team_a, "Draw", team_b],
            y=[prob_a, prob_draw, prob_b],
            marker_color=['#00ff87', '#ffd700', '#60efff'],
            text=[f"{prob_a:.1f}%", f"{prob_draw:.1f}%", f"{prob_b:.1f}%"],
            textposition="outside",
            textfont=dict(size=16, family='Inter', color='#e2e8f0'),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            yaxis=dict(title="Probability (%)", gridcolor='rgba(255,255,255,0.1)', range=[0, 100]),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=50, r=50, t=20, b=50),
            font=dict(family='Inter'),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ===== SIDE-BY-SIDE COMPARISON =====
        st.markdown("### 📊 Team Comparison")
        
        from src.predict import compare_teams
        comp = compare_teams(team_a, team_b, matches_df, metadata_df, elo_df)
        
        # Build comparison rows
        stat_rows = [
            ("FIFA Ranking", comp.get("fifa_ranking", {}).get(team_a, "?"), comp.get("fifa_ranking", {}).get(team_b, "?"), "lower"),
            ("Elo Rating", comp.get("elo_rating", {}).get(team_a, "?"), comp.get("elo_rating", {}).get(team_b, "?"), "higher"),
            ("Market Value (€B)", comp.get("market_value", {}).get(team_a, "?"), comp.get("market_value", {}).get(team_b, "?"), "higher"),
            ("Avg Squad Age", comp.get("avg_squad_age", {}).get(team_a, "?"), comp.get("avg_squad_age", {}).get(team_b, "?"), "neutral"),
            ("Top-5 League Players", comp.get("top5_league_players", {}).get(team_a, "?"), comp.get("top5_league_players", {}).get(team_b, "?"), "higher"),
            ("WC Appearances", comp.get("wc_appearances", {}).get(team_a, "?"), comp.get("wc_appearances", {}).get(team_b, "?"), "higher"),
            ("WC Best Finish", comp.get("wc_best_finish", {}).get(team_a, "?"), comp.get("wc_best_finish", {}).get(team_b, "?"), "neutral"),
            ("Last 5 Wins", comp.get("last_5_wins", {}).get(team_a, "?"), comp.get("last_5_wins", {}).get(team_b, "?"), "higher"),
            ("Goals Scored (Last 5)", comp.get("last_5_goals_scored", {}).get(team_a, "?"), comp.get("last_5_goals_scored", {}).get(team_b, "?"), "higher"),
            ("Goals Conceded (Last 5)", comp.get("last_5_goals_conceded", {}).get(team_a, "?"), comp.get("last_5_goals_conceded", {}).get(team_b, "?"), "lower"),
        ]
        
        # Display as styled table
        header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
        with header_col1:
            st.markdown(f"**🏠 {team_a}**")
        with header_col2:
            st.markdown("<div style='text-align: center; color: #8892b0;'><b>Statistic</b></div>", unsafe_allow_html=True)
        with header_col3:
            st.markdown(f"<div style='text-align: right;'><b>✈️ {team_b}</b></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        for stat_name, val_a, val_b, better in stat_rows:
            c1, c2, c3 = st.columns([1, 2, 1])
            
            # Determine which is better
            try:
                if better == "higher":
                    a_class = "🟢" if float(val_a) > float(val_b) else ("🔴" if float(val_a) < float(val_b) else "⚪")
                    b_class = "🟢" if float(val_b) > float(val_a) else ("🔴" if float(val_b) < float(val_a) else "⚪")
                elif better == "lower":
                    a_class = "🟢" if float(val_a) < float(val_b) else ("🔴" if float(val_a) > float(val_b) else "⚪")
                    b_class = "🟢" if float(val_b) < float(val_a) else ("🔴" if float(val_b) > float(val_a) else "⚪")
                else:
                    a_class = b_class = "⚪"
            except (ValueError, TypeError):
                a_class = b_class = "⚪"
            
            with c1:
                st.markdown(f"{a_class} **{val_a}**")
            with c2:
                st.markdown(f"<div style='text-align: center; color: #8892b0;'>{stat_name}</div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div style='text-align: right;'>{b_class} <b>{val_b}</b></div>", unsafe_allow_html=True)
        
        # Head to Head
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        h2h = comp.get("head_to_head", {})
        if h2h.get("matches", 0) > 0:
            st.markdown("### 📜 Head-to-Head Record")
            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                st.metric(f"{team_a} Wins", h2h["team_a_wins"])
            with hcol2:
                st.metric("Draws", h2h["draws"])
            with hcol3:
                st.metric(f"{team_b} Wins", h2h["team_b_wins"])
            
            if h2h.get("details"):
                st.markdown("**Recent Matches:**")
                for detail in h2h["details"][:5]:
                    st.markdown(f"- {detail['date']}: {detail['score']} ({detail['tournament']})")
        else:
            st.info("No head-to-head history found between these teams.")

else:
    st.markdown("### 👆 Select two teams above and click **Predict** to see the result!")
    
    # Show some quick matchup suggestions
    st.markdown("#### 🔥 Popular Matchups to Try:")
    suggestions = [
        ("Brazil", "Spain"),
        ("Argentina", "France"),
        ("England", "Germany"),
        ("Netherlands", "Portugal"),
        ("Italy", "Croatia"),
    ]
    for t1, t2 in suggestions:
        st.markdown(f"- **{t1}** vs **{t2}**")
