"""
Team Analysis Page — Deep-Dive into Individual Team Statistics
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

st.set_page_config(page_title="Team Analysis | WC 2026", page_icon="📊", layout="wide")

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
    .team-hero {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        border: 1px solid #2d3561; border-radius: 20px; padding: 2rem; text-align: center;
        margin: 1rem 0;
    }
    .team-name-big { font-size: 2.5rem; font-weight: 800; color: #e2e8f0; }
    .team-group { font-size: 1rem; color: #00ff87; font-weight: 600; }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3561; border-radius: 12px; padding: 1rem; text-align: center;
    }
    .stat-val { font-size: 1.6rem; font-weight: 700; color: #00ff87; }
    .stat-lbl { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.5px; }
    .strength-badge {
        display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;
    }
    .tier-1 { background: linear-gradient(135deg, #ffd700, #ffaa00); color: #000; }
    .tier-2 { background: linear-gradient(135deg, #c0c0c0, #a0a0a0); color: #000; }
    .tier-3 { background: linear-gradient(135deg, #cd7f32, #b87333); color: #fff; }
    .tier-4 { background: rgba(255,255,255,0.1); color: #8892b0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">📊 Team Analysis</div>', unsafe_allow_html=True)
st.markdown("*Deep-dive into any team's stats, form, and tournament history*")
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


@st.cache_data
def get_data():
    from src import data_loader
    teams = data_loader.get_all_wc_teams()
    metadata = data_loader.load_team_metadata()
    return teams, metadata

teams, metadata = get_data()

# Team selector
selected_team = st.selectbox("🏳️ Select a Team", teams, 
                              index=teams.index("Argentina") if "Argentina" in teams else 0)

if selected_team:
    from src.predict import get_team_profile
    
    with st.spinner(f"Loading {selected_team} profile..."):
        profile = get_team_profile(selected_team)
    
    if "error" in profile:
        st.error(profile["error"])
        st.stop()
    
    # ===== TEAM HERO CARD =====
    tier_class = f"tier-{profile['strength_tier']}"
    tier_labels = {1: "⭐ Elite", 2: "🔥 Strong", 3: "💪 Competitive", 4: "🌱 Underdog"}
    tier_label = tier_labels.get(profile["strength_tier"], "Unknown")
    
    st.markdown(f"""
    <div class="team-hero">
        <div class="team-name-big">{selected_team}</div>
        <div class="team-group">Group {profile['group']} • {profile['confederation']}</div>
        <div style="margin-top: 0.8rem;">
            <span class="strength-badge {tier_class}">{tier_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== KEY STATS =====
    cols = st.columns(6)
    stats = [
        ("FIFA Rank", f"#{profile['fifa_rank']}"),
        ("Elo Rating", str(profile['elo_rating'])),
        ("Market Value", f"€{profile['market_value_b']}B"),
        ("Avg Age", str(profile['avg_squad_age'])),
        ("WC Apps", str(profile['wc_appearances'])),
        ("Best Finish", profile['wc_best_finish']),
    ]
    
    for col, (label, value) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{value}</div>
                <div class="stat-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== RADAR CHART =====
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🕸️ Team Radar Profile")
        
        radar = profile["radar"]
        categories = list(radar.keys())
        values = list(radar.values())
        
        # Close the radar
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(0, 255, 135, 0.15)',
            line=dict(color='#00ff87', width=2),
            marker=dict(size=8, color='#00ff87'),
            name=selected_team,
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor='rgba(255,255,255,0.1)',
                    tickfont=dict(size=10, color='#8892b0'),
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickfont=dict(size=12, color='#e2e8f0', family='Inter'),
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=60, r=60, t=30, b=30),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Recent Form")
        
        form = profile["form_10"]
        
        # Form stats display
        form_data = {
            "Matches Played": form.get("matches_played_last_10", 0),
            "Wins": form.get("wins_last_10", 0),
            "Draws": form.get("draws_last_10", 0),
            "Losses": form.get("losses_last_10", 0),
            "Win Rate": f"{form.get('win_rate_last_10', 0):.0%}",
            "Goals Scored": form.get("goals_scored_last_10", 0),
            "Goals Conceded": form.get("goals_conceded_last_10", 0),
            "Goal Difference": form.get("goal_diff_last_10", 0),
            "Clean Sheets": form.get("clean_sheets_last_10", 0),
            "Clean Sheet %": f"{form.get('clean_sheet_pct_last_10', 0):.0%}",
        }
        
        for stat_name, stat_value in form_data.items():
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**{stat_name}**")
            with c2:
                st.markdown(f"`{stat_value}`")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== ATTACK VS DEFENSE =====
    st.markdown("### ⚔️ Attack vs Defense Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attack gauge
        attack_score = profile["radar"]["Attack"]
        fig_attack = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attack_score,
            title={'text': "Attack Rating", 'font': {'size': 16, 'color': '#e2e8f0', 'family': 'Inter'}},
            number={'font': {'size': 36, 'color': '#00ff87', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8892b0'},
                'bar': {'color': '#00ff87'},
                'bgcolor': '#1a1a2e',
                'bordercolor': '#2d3561',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(255, 71, 87, 0.2)'},
                    {'range': [33, 66], 'color': 'rgba(255, 215, 0, 0.2)'},
                    {'range': [66, 100], 'color': 'rgba(0, 255, 135, 0.2)'},
                ],
            }
        ))
        fig_attack.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=30, r=30, t=50, b=10),
        )
        st.plotly_chart(fig_attack, use_container_width=True)
    
    with col2:
        # Defense gauge
        defense_score = profile["radar"]["Defense"]
        fig_defense = go.Figure(go.Indicator(
            mode="gauge+number",
            value=defense_score,
            title={'text': "Defense Rating", 'font': {'size': 16, 'color': '#e2e8f0', 'family': 'Inter'}},
            number={'font': {'size': 36, 'color': '#60efff', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8892b0'},
                'bar': {'color': '#60efff'},
                'bgcolor': '#1a1a2e',
                'bordercolor': '#2d3561',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(255, 71, 87, 0.2)'},
                    {'range': [33, 66], 'color': 'rgba(255, 215, 0, 0.2)'},
                    {'range': [66, 100], 'color': 'rgba(0, 255, 135, 0.2)'},
                ],
            }
        ))
        fig_defense.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=30, r=30, t=50, b=10),
        )
        st.plotly_chart(fig_defense, use_container_width=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ===== WORLD CUP HISTORY =====
    st.markdown("### 🏆 World Cup History")
    
    wc = profile["wc_stats"]
    col1, col2, col3, col4 = st.columns(4)
    
    wc_stats = [
        ("Total WC Matches", wc.get("wc_total_matches", 0)),
        ("WC Wins", wc.get("wc_total_wins", 0)),
        ("WC Win Rate", f"{wc.get('wc_win_rate', 0):.0%}"),
        ("WC Goals", wc.get("wc_total_goals", 0)),
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4], wc_stats):
        with col:
            st.metric(label, value)
    
    # ===== INJURIES =====
    injuries = profile.get("injuries", {})
    if injuries.get("injured_key_players", 0) > 0:
        st.markdown("### 🏥 Key Player Injuries")
        st.warning(f"**{injuries['injured_key_players']}** key player(s) injured")
        for detail in injuries.get("details", []):
            st.markdown(f"- ❌ {detail}")
    
    # ===== GROUP RIVALS =====
    st.markdown(f"### 👥 Group {profile['group']} Rivals")
    
    from src import data_loader
    groups = data_loader.load_wc_groups()
    group_teams = groups.get(profile["group"], [])
    
    rival_cols = st.columns(len(group_teams))
    for col, rival in zip(rival_cols, group_teams):
        rival_meta = metadata[metadata["team"] == rival]
        if len(rival_meta) > 0:
            rm = rival_meta.iloc[0]
            is_selected = "🔵" if rival == selected_team else ""
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0;">{is_selected} {rival}</div>
                    <div style="color: #8892b0;">Rank #{rm['fifa_rank']} • Elo {rm['elo_rating']}</div>
                </div>
                """, unsafe_allow_html=True)
