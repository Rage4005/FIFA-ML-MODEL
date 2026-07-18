"""
Reusable team card component for the dashboard.
"""


def render_team_card_html(team_name, fifa_rank, elo_rating, group, market_value=None, 
                           win_probability=None, strength_tier=1):
    """
    Generate HTML for a team card component.
    
    Returns HTML string that can be rendered with st.markdown(html, unsafe_allow_html=True)
    """
    tier_colors = {
        1: ("linear-gradient(135deg, #ffd700, #ffaa00)", "#000"),
        2: ("linear-gradient(135deg, #c0c0c0, #a0a0a0)", "#000"),
        3: ("linear-gradient(135deg, #cd7f32, #b87333)", "#fff"),
        4: ("rgba(255,255,255,0.1)", "#8892b0"),
    }
    tier_labels = {1: "⭐ Elite", 2: "🔥 Strong", 3: "💪 Competitive", 4: "🌱 Underdog"}
    
    bg, fg = tier_colors.get(strength_tier, tier_colors[4])
    label = tier_labels.get(strength_tier, "Unknown")
    
    win_section = ""
    if win_probability is not None:
        win_section = f"""
        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #2d3561;">
            <span style="color: #00ff87; font-size: 1.4rem; font-weight: 700;">{win_probability:.1f}%</span>
            <span style="color: #8892b0; font-size: 0.75rem;"> Win Probability</span>
        </div>
        """
    
    mv_section = ""
    if market_value is not None:
        mv_section = f'<span style="color: #8892b0; font-size: 0.8rem;"> • €{market_value}B</span>'
    
    html = f"""
    <div style="background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
                border: 1px solid #2d3561; border-radius: 16px; padding: 1.2rem;
                transition: transform 0.2s ease; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.3rem; font-weight: 700; color: #e2e8f0;">{team_name}</span>
                <span style="font-size: 0.8rem; color: #8892b0; margin-left: 8px;">Group {group}</span>
                {mv_section}
            </div>
            <span style="padding: 3px 12px; border-radius: 12px; background: {bg}; 
                         color: {fg}; font-size: 0.75rem; font-weight: 600;">{label}</span>
        </div>
        <div style="margin-top: 0.5rem; display: flex; gap: 1.5rem;">
            <div>
                <span style="color: #8892b0; font-size: 0.75rem;">FIFA Rank</span><br>
                <span style="color: #e2e8f0; font-weight: 600;">#{fifa_rank}</span>
            </div>
            <div>
                <span style="color: #8892b0; font-size: 0.75rem;">Elo Rating</span><br>
                <span style="color: #e2e8f0; font-weight: 600;">{elo_rating}</span>
            </div>
        </div>
        {win_section}
    </div>
    """
    
    return html
