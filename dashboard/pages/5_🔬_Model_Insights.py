"""
Model Insights Page — Feature Importance, Accuracy, and Diagnostics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Model Insights | WC 2026", page_icon="🔬", layout="wide")

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
    .insight-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3561; border-radius: 16px; padding: 1.5rem; text-align: center;
    }
    .insight-value { font-size: 2rem; font-weight: 700; color: #00ff87; }
    .insight-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔬 Model Insights</div>', unsafe_allow_html=True)
st.markdown("*Understand how the prediction model works and how accurate it is*")
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# Try to load model metadata
model_loaded = False
model_meta = None

try:
    from src.model import load_model, MODELS_DIR
    model, model_meta = load_model("xgb_match_predictor")
    model_loaded = True
except Exception as e:
    pass

if not model_loaded:
    st.warning("⚠️ Model not trained yet. Run `python scripts/train_model.py` first to see model insights.")
    
    st.markdown("""
    ### 📖 About the Model
    
    This project uses an **XGBoost** gradient boosting classifier to predict football match outcomes.
    
    #### How It Works:
    
    1. **Data Collection**: Historical international match results (10,000+ matches from 2010-2026)
    2. **Feature Engineering**: 50+ features including:
       - Team strength (Elo rating, FIFA ranking, market value)
       - Recent form (rolling wins, goals, clean sheets over last 5/10 matches)
       - Attacking & defensive performance metrics
       - World Cup historical performance
       - Squad composition (age, top-league players)
    3. **Model Training**: XGBoost multi-class classification
       - Classes: Home Win (0), Draw (1), Away Win (2)
       - Hyperparameters optimized for football prediction
    4. **Tournament Simulation**: Monte Carlo method
       - Each match simulated using model probabilities
       - Full tournament run 10,000 times
       - Win probability = wins / simulations
    
    #### Expected Performance:
    - **Accuracy**: ~50-60% (3-class is hard — random baseline is 33%)
    - **Key Features**: Elo rating difference, recent form, FIFA ranking
    - **Limitation**: Football is inherently unpredictable — upsets happen!
    
    Run the training script to see actual metrics:
    ```bash
    python scripts/train_model.py
    ```
    """)
    st.stop()

# ===== MODEL IS LOADED — SHOW INSIGHTS =====

# Key metrics
col1, col2, col3, col4 = st.columns(4)

accuracy = model_meta.get("accuracy", 0) if model_meta else 0
f1 = model_meta.get("f1_macro", 0) if model_meta else 0
log_loss_val = model_meta.get("log_loss", 0) if model_meta else 0
cv_mean = model_meta.get("cv_mean", 0) if model_meta else 0

with col1:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-value">{accuracy:.1%}</div>
        <div class="insight-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-value">{f1:.3f}</div>
        <div class="insight-label">F1 Score (Macro)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-value">{log_loss_val:.3f}</div>
        <div class="insight-label">Log Loss</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-value">{cv_mean:.1%}</div>
        <div class="insight-label">CV Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ===== FEATURE IMPORTANCE =====
st.markdown("### 🏅 Feature Importance (Top 15)")

if model_meta and "top_features" in model_meta:
    top_features = model_meta["top_features"]
    
    feat_names = [f[0] for f in reversed(top_features)]
    feat_vals = [f[1] for f in reversed(top_features)]
    
    # Color gradient
    n = len(feat_names)
    colors = [f'rgba(0, {int(255 * (i/n))}, {int(135 * (i/n))}, 0.8)' for i in range(n)]
    colors = [f'hsl({120 + i * 8}, 80%, {40 + i * 3}%)' for i in range(n)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feat_names,
        x=feat_vals,
        orientation='h',
        marker=dict(
            color=feat_vals,
            colorscale=[[0, '#16213e'], [0.5, '#00cc6a'], [1, '#00ff87']],
        ),
        text=[f" {v:.4f}" for v in feat_vals],
        textposition="outside",
        textfont=dict(size=12, color='#e2e8f0', family='Inter'),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        xaxis=dict(title="Importance Score", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=0, r=100, t=10, b=50),
        font=dict(family='Inter', size=12),
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance data not available in model metadata.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ===== CONFUSION MATRIX =====
st.markdown("### 📊 Confusion Matrix")

if model_meta and "confusion_matrix" in model_meta:
    cm = np.array(model_meta["confusion_matrix"])
    labels = ["Home Win", "Draw", "Away Win"]
    
    # Normalize
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=16, family='Inter'),
        colorscale=[[0, '#0a0a1a'], [0.5, '#0f3460'], [1, '#00ff87']],
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.1%}<extra></extra>",
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        margin=dict(l=80, r=50, t=30, b=80),
        font=dict(family='Inter', size=13),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Per-class metrics
    if "classification_report" in model_meta:
        report = model_meta["classification_report"]
        st.markdown("#### Per-Class Metrics")
        
        report_data = []
        for class_name in ["Home Win", "Draw", "Away Win"]:
            if class_name in report:
                cls = report[class_name]
                report_data.append({
                    "Class": class_name,
                    "Precision": f"{cls.get('precision', 0):.3f}",
                    "Recall": f"{cls.get('recall', 0):.3f}",
                    "F1-Score": f"{cls.get('f1-score', 0):.3f}",
                    "Support": cls.get("support", 0),
                })
        
        if report_data:
            st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)
else:
    st.info("Confusion matrix not available. Train the model first.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ===== MODEL DETAILS =====
st.markdown("### ⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Training Details")
    n_features = model_meta.get("n_features", "?") if model_meta else "?"
    n_samples = model_meta.get("n_training_samples", "?") if model_meta else "?"
    cv_std = model_meta.get("cv_std", 0) if model_meta else 0
    
    details = {
        "Algorithm": "XGBoost (multi:softprob)",
        "Classes": "Home Win / Draw / Away Win",
        "Features": str(n_features),
        "Training Samples": str(n_samples),
        "CV Mean ± Std": f"{cv_mean:.4f} ± {cv_std:.4f}",
        "Test Accuracy": f"{accuracy:.4f}",
    }
    
    for key, val in details.items():
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**{key}**")
        with c2:
            st.markdown(f"`{val}`")

with col2:
    st.markdown("#### XGBoost Hyperparameters")
    
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    
    for key, val in params.items():
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**{key}**")
        with c2:
            st.markdown(f"`{val}`")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ===== INTERPRETATION GUIDE =====
with st.expander("📖 How to Interpret These Metrics"):
    st.markdown("""
    ### Understanding the Model
    
    | Metric | What It Means | Good Value |
    |--------|---------------|------------|
    | **Accuracy** | % of correct predictions | >50% (baseline is 33%) |
    | **F1 Score** | Balance of precision & recall | >0.45 |
    | **Log Loss** | Confidence calibration | <1.0 |
    | **CV Accuracy** | Consistency across data splits | Close to test accuracy |
    
    ### About Football Prediction
    
    - **33% is random chance** (3 classes: win/draw/loss)
    - **50-55% is considered good** for football match prediction
    - **60%+ is excellent** and approaches the ceiling for this task
    - **Draws are the hardest to predict** — models typically struggle here
    - **Elo rating is usually the strongest feature** — it captures team strength well
    
    ### Important Caveats
    
    ⚠️ Football is inherently stochastic. Even the best models can't predict:
    - Last-minute goals
    - Red cards and penalties
    - Tactical surprises
    - Player injuries during the match
    - Emotional/psychological factors
    
    The model gives **probabilities**, not certainties. Use them to understand *likely* outcomes.
    """)
