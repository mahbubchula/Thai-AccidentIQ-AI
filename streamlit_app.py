"""
Thai Road Accident Analysis - AI-Powered Dashboard
Modern, beautiful interface with AI theme
Author: MAHBUB Hassan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Page config
st.set_page_config(
    page_title="Thai Road Safety AI | MAHBUB Hassan",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern AI theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Override for better readability */
    .main {
        background: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem;
        max-width: 1400px;
        background: white;
    }
    
    /* Headers */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #2d2d44 !important;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #2d2d44 !important;
        font-size: 2rem !important;
        margin-top: 2rem;
    }
    
    h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #4a4a6a !important;
        font-size: 1.5rem !important;
    }
    
    /* Regular text */
    p, li, span, div {
        color: #2d2d44 !important;
    }
    
    /* Cards with glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #2d2d44 !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        color: #4a4a6a !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 50px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #4a4a6a;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        border-radius: 50px;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    return df

@st.cache_data
def load_results():
    results = {}
    
    # Feature importance
    if os.path.exists(os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')):
        results['importance'] = pd.read_csv(os.path.join(RESULTS_DIR, 'shap_feature_importance.csv'))
    
    # Model comparison
    if os.path.exists(os.path.join(RESULTS_DIR, 'model_comparison.csv')):
        results['models'] = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
    
    # LLM insights
    if os.path.exists(os.path.join(REPORTS_DIR, 'llm_insights.json')):
        with open(os.path.join(REPORTS_DIR, 'llm_insights.json'), 'r', encoding='utf-8') as f:
            results['llm'] = json.load(f)
    
    return results

# Hero Section
st.markdown("""
<div style='text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 2rem;'>
    <h1 style='font-size: 4rem; margin-bottom: 1rem; color: white !important;'>🚗 Thai Road Safety AI</h1>
    <p style='font-size: 1.5rem; color: white !important; font-weight: 300;'>
        Advanced Machine Learning for Accident Prevention
    </p>
    <p style='font-size: 1.1rem; color: rgba(255, 255, 255, 0.9) !important; margin-top: 1rem;'>
        Powered by XGBoost + SHAP + Llama 3.3 70B | 81,735 Accidents Analyzed
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
try:
    df = load_data()
    results = load_results()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:
    # Key Metrics
    st.markdown("<div class='animated'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Accidents",
            value=f"{len(df):,}",
            delta="2019-2022"
        )
    
    with col2:
        fatal_pct = (df['severity_class'] == 'fatal').sum() / len(df) * 100
        st.metric(
            label="⚠️ Fatal Rate",
            value=f"{fatal_pct:.1f}%",
            delta=f"{(df['severity_class'] == 'fatal').sum():,} deaths"
        )
    
    with col3:
        if 'models' in results:
            best_f1 = results['models'].iloc[0]['F1-Score']
            st.metric(
                label="🎯 Model F1-Score",
                value=f"{best_f1:.3f}",
                delta="XGBoost Best"
            )
    
    with col4:
        high_severity = (df['high_severity'] == 1).sum()
        st.metric(
            label="🚨 High Severity",
            value=f"{high_severity:,}",
            delta=f"{high_severity/len(df)*100:.1f}%"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tabs for different sections
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔮 Predict",
        "🧠 AI Insights",
        "📄 Reports"
    ])
    
    with tab1:
        st.markdown("## 📊 Interactive Dashboard")
        
        # Temporal Analysis
        st.markdown("### 📅 Temporal Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Accidents by hour
            hourly = df.groupby('hour').size().reset_index(name='count')
            fig = px.bar(
                hourly, x='hour', y='count',
                title='Accidents by Hour of Day',
                color='count',
                color_continuous_scale='Purples',
                labels={'hour': 'Hour', 'count': 'Number of Accidents'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                title_font=dict(size=18, family='Space Grotesk', color='#2d2d44'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accidents by day of week
            dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            dow_df = df.groupby('day_of_week').size().reset_index(name='count')
            dow_df['day'] = dow_df['day_of_week'].map(dow_map)
            
            fig = px.bar(
                dow_df, x='day', y='count',
                title='Accidents by Day of Week',
                color='count',
                color_continuous_scale='Purples',
                labels={'day': 'Day', 'count': 'Number of Accidents'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                title_font=dict(size=18, family='Space Grotesk', color='#2d2d44'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Severity Analysis
        st.markdown("### 🎯 Severity Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            severity_counts = df['severity_class'].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title='Severity Distribution',
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.4
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                title_font=dict(size=18, family='Space Grotesk', color='#2d2d44')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            vehicle_severity = df.groupby('vehicle_type')['high_severity'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=vehicle_severity.values * 100,
                y=vehicle_severity.index,
                orientation='h',
                title='High Severity Rate by Vehicle Type (Top 10)',
                color=vehicle_severity.values,
                color_continuous_scale='Purples',
                labels={'x': 'High Severity Rate (%)', 'y': 'Vehicle Type'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                title_font=dict(size=18, family='Space Grotesk', color='#2d2d44'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if 'importance' in results:
            st.markdown("### 🔍 Feature Importance (SHAP)")
            top_features = results['importance'].head(10)
            
            fig = px.bar(
                top_features.sort_values('importance'),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='importance',
                color_continuous_scale='Purples',
                labels={'importance': 'SHAP Value', 'feature': 'Feature'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                title_font=dict(size=18, family='Space Grotesk', color='#2d2d44'),
                showlegend=False,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## 🔮 Accident Severity Prediction")
        st.info("⚠️ Note: This is a demonstration interface. Full prediction functionality requires loading the trained model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Input Accident Details")
            
            vehicle_type = st.selectbox(
                "Vehicle Type",
                options=sorted(df['vehicle_type'].unique())
            )
            
            accident_type = st.selectbox(
                "Accident Type",
                options=sorted(df['accident_type'].unique())
            )
            
            num_vehicles = st.number_input(
                "Number of Vehicles Involved",
                min_value=1, max_value=10, value=1
            )
            
            weather = st.selectbox(
                "Weather Condition",
                options=sorted(df['weather_condition'].unique())
            )
        
        with col2:
            st.markdown("### Additional Information")
            
            hour = st.slider("Hour of Day", 0, 23, 12)
            
            road_desc = st.selectbox(
                "Road Description",
                options=sorted(df['road_description'].unique())
            )
            
            slope = st.selectbox(
                "Slope Description",
                options=sorted(df['slope_description'].unique())
            )
        
        if st.button("🔮 Predict Severity", use_container_width=True):
            st.success("✅ Prediction functionality coming soon!")
            st.markdown("""
            **Demo Output:**
            - Predicted Severity: High
            - Confidence: 73%
            - Top Risk Factors: Vehicle Type, Weather Condition
            """)
    
    with tab3:
        st.markdown("## 🧠 AI-Powered Insights")
        
        if 'llm' in results:
            # Executive Summary
            st.markdown("### 📋 Executive Summary")
            with st.expander("View Full Summary", expanded=True):
                st.markdown(results['llm']['executive_summary'])
            
            # Safety Recommendations
            st.markdown("### 🚦 Safety Recommendations")
            with st.expander("View Recommendations", expanded=True):
                st.markdown(results['llm']['safety_recommendations'])
            
            # Q&A Examples
            st.markdown("### 💬 Ask the AI")
            
            if 'qa_examples' in results['llm']:
                for question, answer in list(results['llm']['qa_examples'].items())[:2]:
                    with st.chat_message("user"):
                        st.write(question)
                    with st.chat_message("assistant"):
                        st.write(answer)
        else:
            st.warning("LLM insights not available. Run Step 5 to generate insights.")
    
    with tab4:
        st.markdown("## 📄 Research Reports")
        
        st.markdown("### 📥 Available Downloads")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='glass-card'>
                <h3>📊 Model Performance</h3>
                <p>Detailed comparison of all ML models with metrics</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(os.path.join(RESULTS_DIR, 'model_comparison.csv')):
                with open(os.path.join(RESULTS_DIR, 'model_comparison.csv'), 'r') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f.read(),
                        file_name="model_comparison.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col2:
            st.markdown("""
            <div class='glass-card'>
                <h3>🔍 SHAP Analysis</h3>
                <p>Feature importance rankings and explanations</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')):
                with open(os.path.join(RESULTS_DIR, 'shap_feature_importance.csv'), 'r') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f.read(),
                        file_name="shap_feature_importance.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col3:
            st.markdown("""
            <div class='glass-card'>
                <h3>🤖 LLM Insights</h3>
                <p>AI-generated policy recommendations</p>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(os.path.join(REPORTS_DIR, 'llm_insights_report.txt')):
                with open(os.path.join(REPORTS_DIR, 'llm_insights_report.txt'), 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="Download TXT",
                        data=f.read(),
                        file_name="llm_insights_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: white; font-size: 1.5rem;'>🎓 Research Info</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='color: rgba(255,255,255,0.9); margin: 0;'><b>Author:</b> MAHBUB Hassan</p>
        <p style='color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
            PhD Student<br>
            Transportation Engineering<br>
            Chulalongkorn University
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 Methodology")
    st.markdown("""
    - **Dataset**: 81,735 accidents
    - **Timeframe**: 2019-2022
    - **Models**: XGBoost, LightGBM, CatBoost
    - **XAI**: SHAP Analysis
    - **LLM**: Llama 3.3 70B
    """)
    
    st.markdown("### 📊 Key Metrics")
    if 'models' in results:
        best_model = results['models'].iloc[0]
        st.markdown(f"""
        - **F1-Score**: {best_model['F1-Score']:.3f}
        - **ROC-AUC**: {best_model['ROC-AUC']:.3f}
        - **Precision**: {best_model['Precision']:.3f}
        - **Recall**: {best_model['Recall']:.3f}
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: rgba(255,255,255,0.5); font-size: 0.8rem;'>
            © 2025 MAHBUB Hassan<br>
            All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p><b>Thai Road Safety AI Dashboard</b></p>
    <p>Powered by Machine Learning, Explainable AI, and Large Language Models</p>
    <p>Built with Streamlit | Data: Thai Department of Highways (2019-2022)</p>
</div>
""", unsafe_allow_html=True)