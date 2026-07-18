"""
Reusable Plotly chart components for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# Shared theme settings
DARK_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font_family": "Inter",
    "grid_color": "rgba(255,255,255,0.1)",
    "accent_green": "#00ff87",
    "accent_cyan": "#60efff",
    "accent_gold": "#ffd700",
    "accent_red": "#ff4757",
    "text_primary": "#e2e8f0",
    "text_secondary": "#8892b0",
}


def create_probability_bar(team_a, prob_a, prob_draw, team_b, prob_b, height=300):
    """Create a vertical bar chart for match probabilities."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[team_a, "Draw", team_b],
        y=[prob_a, prob_draw, prob_b],
        marker_color=[DARK_THEME["accent_green"], DARK_THEME["accent_gold"], DARK_THEME["accent_cyan"]],
        text=[f"{prob_a:.1f}%", f"{prob_draw:.1f}%", f"{prob_b:.1f}%"],
        textposition="outside",
        textfont=dict(size=16, family='Inter', color=DARK_THEME["text_primary"]),
    ))
    
    fig.update_layout(
        template=DARK_THEME["template"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["plot_bgcolor"],
        height=height,
        yaxis=dict(title="Probability (%)", gridcolor=DARK_THEME["grid_color"], range=[0, 100]),
        margin=dict(l=50, r=50, t=20, b=50),
        font=dict(family=DARK_THEME["font_family"]),
    )
    
    return fig


def create_radar_chart(categories, values, team_name, height=400):
    """Create a radar/spider chart for team profile."""
    # Close the polygon
    categories_closed = list(categories) + [categories[0]]
    values_closed = list(values) + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=f'rgba(0, 255, 135, 0.15)',
        line=dict(color=DARK_THEME["accent_green"], width=2),
        marker=dict(size=8, color=DARK_THEME["accent_green"]),
        name=team_name,
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=DARK_THEME["grid_color"]),
            angularaxis=dict(gridcolor=DARK_THEME["grid_color"]),
            bgcolor='rgba(0,0,0,0)',
        ),
        template=DARK_THEME["template"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        height=height,
        showlegend=False,
    )
    
    return fig


def create_horizontal_bar(teams, values, title="", height=400, color=None):
    """Create a horizontal bar chart for rankings/probabilities."""
    if color is None:
        color = DARK_THEME["accent_green"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=teams,
        x=values,
        orientation='h',
        marker_color=color,
        text=[f" {v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=12, color=DARK_THEME["text_primary"], family='Inter'),
    ))
    
    fig.update_layout(
        template=DARK_THEME["template"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["plot_bgcolor"],
        height=height,
        xaxis=dict(title=title, gridcolor=DARK_THEME["grid_color"]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=0, r=80, t=10, b=50),
        font=dict(family=DARK_THEME["font_family"], size=12),
    )
    
    return fig


def create_gauge(value, title, max_val=100, color=None, height=250):
    """Create a gauge/speedometer chart."""
    if color is None:
        color = DARK_THEME["accent_green"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': DARK_THEME["text_primary"]}},
        number={'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': color},
            'bgcolor': '#1a1a2e',
            'bordercolor': '#2d3561',
            'steps': [
                {'range': [0, max_val * 0.33], 'color': 'rgba(255, 71, 87, 0.2)'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': 'rgba(255, 215, 0, 0.2)'},
                {'range': [max_val * 0.66, max_val], 'color': 'rgba(0, 255, 135, 0.2)'},
            ],
        }
    ))
    
    fig.update_layout(
        template=DARK_THEME["template"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        height=height,
        margin=dict(l=30, r=30, t=50, b=10),
    )
    
    return fig


def create_heatmap(z, x_labels, y_labels, title="", height=400):
    """Create a heatmap (e.g., for confusion matrix)."""
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, '#0a0a1a'], [0.5, '#0f3460'], [1, DARK_THEME["accent_green"]]],
        text=z,
        texttemplate="%{text}",
        textfont=dict(size=14, family='Inter'),
    ))
    
    fig.update_layout(
        template=DARK_THEME["template"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["plot_bgcolor"],
        height=height,
        title=title,
        margin=dict(l=80, r=50, t=50, b=80),
        font=dict(family=DARK_THEME["font_family"]),
    )
    
    return fig
