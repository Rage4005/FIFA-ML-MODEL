"""
FIFA World Cup 2026 Prediction Dashboard — Main App

Streamlit entry point with navigation and global configuration.
"""

import streamlit as st
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    st.set_page_config(
        page_title="⚽ FIFA World Cup 2026 Predictor",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for premium look
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global styles */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Dark theme enhancement */
        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }
        
        /* Hero section */
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
            line-height: 1.2;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
            color: #8892b0;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #2d3561;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 255, 135, 0.15);
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #00ff87;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.3rem;
        }
        
        /* Team selector */
        .team-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border: 1px solid #2d3561;
            border-radius: 12px;
            margin: 0.25rem;
            font-weight: 500;
            color: #e2e8f0;
        }
        
        /* Probability bar */
        .prob-container {
            background: #1a1a2e;
            border-radius: 12px;
            overflow: hidden;
            margin: 0.5rem 0;
            height: 36px;
            position: relative;
        }
        
        .prob-bar {
            height: 100%;
            border-radius: 12px;
            display: flex;
            align-items: center;
            padding-left: 12px;
            font-weight: 600;
            font-size: 0.85rem;
            transition: width 0.8s ease;
        }
        
        .prob-win { background: linear-gradient(90deg, #00ff87, #00cc6a); color: #000; }
        .prob-draw { background: linear-gradient(90deg, #ffd700, #ffaa00); color: #000; }
        .prob-lose { background: linear-gradient(90deg, #ff4757, #ff2e44); color: #fff; }
        
        /* Comparison table */
        .comparison-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #1e2a3a;
        }
        
        .comparison-row:hover {
            background: rgba(0, 255, 135, 0.05);
        }
        
        .stat-value-left {
            font-weight: 600;
            font-size: 1rem;
            min-width: 80px;
            text-align: left;
        }
        
        .stat-name {
            font-size: 0.85rem;
            color: #8892b0;
            text-align: center;
            flex: 1;
        }
        
        .stat-value-right {
            font-weight: 600;
            font-size: 1rem;
            min-width: 80px;
            text-align: right;
        }
        
        .better { color: #00ff87; }
        .worse { color: #ff4757; }
        .neutral { color: #e2e8f0; }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0a1a 0%, #0f1629 100%);
        }
        
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        /* Hide default Streamlit menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Divider */
        .gradient-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #00ff87, #60efff, transparent);
            margin: 2rem 0;
            border: none;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏆 World Cup 2026")
        st.markdown("### AI Prediction Engine")
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Powered by:**
        - 🤖 XGBoost ML Model
        - 📊 10,000+ Match History  
        - 🎲 Monte Carlo Simulation
        - ⚽ 48 Team Analysis
        """)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 📋 Quick Links")
        st.page_link("pages/1_🏠_Home.py", label="🏠 Home & Favorites")
        st.page_link("pages/2_⚽_Match_Predictor.py", label="⚽ Match Predictor")
        st.page_link("pages/3_🏆_Tournament_Sim.py", label="🏆 Tournament Simulator")
        st.page_link("pages/4_📊_Team_Analysis.py", label="📊 Team Analysis")
        st.page_link("pages/5_🔬_Model_Insights.py", label="🔬 Model Insights")
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.caption("Built with ❤️ for FIFA World Cup 2026")
    
    # Main content - redirect to Home
    st.markdown('<div class="hero-title">⚽ FIFA World Cup 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-Powered Prediction Engine • Machine Learning • Monte Carlo Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.info("👈 Use the sidebar to navigate between pages, or click the links above!")
    
    # Quick overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">48</div>
            <div class="metric-label">Teams</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12</div>
            <div class="metric-label">Groups</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">104</div>
            <div class="metric-label">Matches</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">10K+</div>
            <div class="metric-label">Training Data</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
