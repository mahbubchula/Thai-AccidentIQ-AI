"""
Thai AccidentIQ AI - Complete Production Version
Real-time accident severity prediction with AI chat
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
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from groq import Groq

# Page config
st.set_page_config(
    page_title="Thai AccidentIQ AI | MAHBUB Hassan",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern, Readable
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .stApp {
        background: white;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .main .block-container {
        padding: 2rem;
        max-width: 1400px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #2d2d44 !important;
    }
    
    h1 { font-size: 3rem !important; font-weight: 700; }
    h2 { font-size: 2rem !important; font-weight: 600; margin-top: 2rem; }
    h3 { font-size: 1.5rem !important; font-weight: 600; }
    
    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .hero-banner h1 {
        color: white !important;
        font-size: 4rem !important;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #2d2d44 !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 50px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
    }
    
    .prediction-box h2 {
        color: white !important;
        margin-top: 0;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer h3, .footer p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "best_tuned_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

# Initialize Groq (for chat)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load data and model
@st.cache_resource
def load_model_and_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        
        # Get feature info
        feature_cols = [col for col in df.columns if col not in [
            'acc_code', 'route', 'province_th', 'incident_datetime', 'report_datetime',
            'severity_class', 'high_severity', 'total_casualties', 
            'number_of_fatalities', 'number_of_injuries'
        ]]
        
        return df, model, scaler, feature_cols, True
    except Exception as e:
        st.error(f"Error loading: {e}")
        return None, None, None, None, False

df, model, scaler, feature_cols, loaded = load_model_and_data()

# Hero Section
st.markdown("""
<div class='hero-banner'>
    <h1>🚗 Thai AccidentIQ AI</h1>
    <p style='font-size: 1.5rem; margin-bottom: 0.5rem;'>
        Intelligent Road Safety Analytics & Prediction System
    </p>
    <p style='font-size: 1rem; opacity: 0.9;'>
        Powered by XGBoost + SHAP + Llama 3.3 70B | 81,735 Accidents Analyzed
    </p>
</div>
""", unsafe_allow_html=True)

if loaded:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Accidents", f"{len(df):,}", "2019-2022")
    
    with col2:
        fatal_pct = (df['severity_class'] == 'fatal').sum() / len(df) * 100
        st.metric("⚠️ Fatal Rate", f"{fatal_pct:.1f}%", f"{(df['severity_class'] == 'fatal').sum():,} deaths")
    
    with col3:
        st.metric("🎯 Model F1-Score", "0.523", "XGBoost Optimized")
    
    with col4:
        high_severity = (df['high_severity'] == 1).sum()
        st.metric("🚨 High Severity", f"{high_severity:,}", f"{high_severity/len(df)*100:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔮 AI Prediction",
        "💬 AI Chat Assistant",
        "🧠 Insights",
        "ℹ️ About"
    ])
    
    with tab1:
        st.markdown("## 📊 Interactive Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly = df.groupby('hour').size().reset_index(name='count')
            fig = px.bar(hourly, x='hour', y='count',
                        title='🕐 Accidents by Hour of Day',
                        color='count', color_continuous_scale='Purples')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Severity distribution
            severity_counts = df['severity_class'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title='🎯 Severity Distribution',
                        color_discrete_sequence=px.colors.sequential.Purples_r, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week
            dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            dow_df = df.groupby('day_of_week').size().reset_index(name='count')
            dow_df['day'] = dow_df['day_of_week'].map(dow_map)
            fig = px.bar(dow_df, x='day', y='count',
                        title='📅 Accidents by Day of Week',
                        color='count', color_continuous_scale='Purples')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Vehicle risk
            vehicle_severity = df.groupby('vehicle_type')['high_severity'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=vehicle_severity.values * 100, y=vehicle_severity.index, orientation='h',
                        title='🚙 High Severity Rate by Vehicle Type',
                        color=vehicle_severity.values, color_continuous_scale='Purples')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## 🔮 Real-Time Accident Severity Prediction")
        
        st.info("⚡ Enter accident details below to get AI-powered severity prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🚗 Vehicle & Accident")
            vehicle_type = st.selectbox("Vehicle Type", options=sorted(df['vehicle_type'].unique()))
            accident_type = st.selectbox("Accident Type", options=sorted(df['accident_type'].unique()))
            num_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=10, value=1)
        
        with col2:
            st.markdown("### 🌤️ Conditions")
            weather = st.selectbox("Weather", options=sorted(df['weather_condition'].unique()))
            road_desc = st.selectbox("Road Type", options=sorted(df['road_description'].unique()))
            slope = st.selectbox("Slope", options=sorted(df['slope_description'].unique()))
        
        with col3:
            st.markdown("### 📍 Location & Time")
            province = st.selectbox("Province", options=sorted(df['province_en'].unique()))
            hour = st.slider("Hour", 0, 23, 12)
            day_of_week = st.selectbox("Day", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            season = st.selectbox("Season", options=['hot', 'rainy', 'cool'])
        
        if st.button("🔮 Predict Severity", use_container_width=True):
            try:
                # Prepare input
                dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                          'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                
                input_data = {
                    'province_en': province,
                    'agency': df['agency'].mode()[0],
                    'vehicle_type': vehicle_type,
                    'presumed_cause': df['presumed_cause'].mode()[0],
                    'accident_type': accident_type,
                    'number_of_vehicles_involved': num_vehicles,
                    'weather_condition': weather,
                    'latitude': df[df['province_en']==province]['latitude'].median(),
                    'longitude': df[df['province_en']==province]['longitude'].median(),
                    'road_description': road_desc,
                    'slope_description': slope,
                    'year': 2024,
                    'month': datetime.now().month,
                    'day': 15,
                    'hour': hour,
                    'day_of_week': dow_map[day_of_week],
                    'is_weekend': 1 if dow_map[day_of_week] >= 5 else 0,
                    'season': season,
                    'time_of_day': 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 17 else 'evening' if 17 <= hour < 21 else 'night',
                    'is_rush_hour': 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
                }
                
                # Encode
                input_df = pd.DataFrame([input_data])
                categorical_cols = input_df.select_dtypes(include=['object']).columns
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    input_df[col] = le.transform(input_df[col].astype(str))
                
                # Scale
                input_scaled = scaler.transform(input_df[feature_cols])
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                severity = "HIGH SEVERITY ⚠️" if prediction == 1 else "LOW SEVERITY ✅"
                confidence = probability[prediction] * 100
                
                # Display result
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2>Prediction Result</h2>
                    <h1 style='color: white !important; font-size: 3rem;'>{severity}</h1>
                    <p style='font-size: 1.5rem;'>Confidence: {confidence:.1f}%</p>
                    <p style='font-size: 1rem; opacity: 0.9;'>High Severity Probability: {probability[1]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors
                st.markdown("### 🎯 Key Risk Factors")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"**Vehicle**: {vehicle_type}")
                with col2:
                    st.info(f"**Accident Type**: {accident_type}")
                with col3:
                    st.info(f"**Vehicles Involved**: {num_vehicles}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with tab3:
        st.markdown("## 💬 AI Chat Assistant")
        st.info("🤖 Ask me anything about Thai road accident patterns, safety recommendations, or data insights!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about road safety, predictions, or data insights..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            try:
                client = Groq(api_key=GROQ_API_KEY)
                
                context = f"""You are Thai AccidentIQ AI, an expert road safety analyst. 
                
                Dataset context:
                - 81,735 Thai road accidents (2019-2022)
                - Top predictor: Vehicle type (SHAP: 0.47)
                - 74% caused by speeding
                - 12.4% fatal rate
                - Model: XGBoost (F1: 0.52, ROC-AUC: 0.81)
                
                Answer concisely and professionally. Use data to support answers.
                """
                
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
            except Exception as e:
                st.error(f"Chat error: {e}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with tab4:
        st.markdown("## 🧠 AI-Generated Insights")
        
        # Load LLM insights
        llm_file = os.path.join(REPORTS_DIR, 'llm_insights.json')
        if os.path.exists(llm_file):
            with open(llm_file, 'r', encoding='utf-8') as f:
                llm = json.load(f)
            
            st.markdown("### 📋 Executive Summary")
            with st.expander("Read Full Summary", expanded=True):
                st.markdown(llm['executive_summary'])
            
            st.markdown("### 🚦 Safety Recommendations")
            with st.expander("View Policy Recommendations"):
                st.markdown(llm['safety_recommendations'])
        
        # Feature importance
        importance_file = os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')
        if os.path.exists(importance_file):
            importance = pd.read_csv(importance_file)
            st.markdown("### 🔍 Top Features (SHAP Analysis)")
            
            top10 = importance.head(10)
            fig = px.bar(top10.sort_values('importance'), x='importance', y='feature',
                        orientation='h', title='Top 10 Predictive Features',
                        color='importance', color_continuous_scale='Purples')
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## ℹ️ About Thai AccidentIQ AI")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            **Thai AccidentIQ AI** is an advanced machine learning system designed to predict and analyze 
            road accident severity in Thailand. Using 81,735 historical accidents from 2019-2022, the system 
            provides real-time predictions, interactive analytics, and AI-powered safety insights.
            
            ### 🔬 Methodology
            
            - **Data**: 81,735 accidents across 78 Thai provinces (2019-2022)
            - **ML Models**: XGBoost, LightGBM, CatBoost (optimized with Optuna)
            - **Best Model**: XGBoost (F1-Score: 0.523, ROC-AUC: 0.81)
            - **Explainability**: SHAP analysis for interpretable predictions
            - **AI Integration**: Llama 3.3 70B for natural language insights
            
            ### 📊 Key Findings
            
            1. **Vehicle type** is the strongest predictor (SHAP: 0.47)
            2. **Speeding** causes 74% of all accidents
            3. **Geographic location** significantly impacts severity
            4. **Multi-vehicle accidents** are 3x more severe
            5. **Pickup trucks** have highest involvement rate (35%)
            
            ### 🎓 Research Impact
            
            This research provides evidence-based insights for:
            - Transportation policy makers
            - Thai Department of Highways
            - Emergency response planning
            - Public safety campaigns
            """)
        
        with col2:
            st.markdown("""
            ### 👨‍🎓 Researcher
            
            **MAHBUB Hassan**
            
            Graduate Student & Non ASEAN Scholar
            
            **Department**: Civil Engineering
            
            **Faculty**: Engineering
            
            **University**: Chulalongkorn University
            
            **Location**: Bangkok, Thailand
            
            **Research Focus**:
            - Transportation Safety
            - Machine Learning Applications
            - Climate-Adaptive Systems
            - AI in Transportation
            
            ---
            
            ### 📧 Contact
            
            For collaboration or inquiries about this research, please contact through Chulalongkorn University.
            
            ---
            
            ### 🔗 Technologies
            
            - Python & Streamlit
            - XGBoost & SHAP
            - Groq & Llama 3.3
            - Plotly & Pandas
            """)

# Footer
st.markdown("""
<div class='footer'>
    <h3>🚗 Thai AccidentIQ AI</h3>
    <p><b>Intelligent Road Safety Analytics & Prediction System</b></p>
    <p style='margin-top: 1rem; opacity: 0.9;'>
        Developed by <b>MAHBUB Hassan</b><br>
        Graduate Student, Department of Civil Engineering<br>
        Chulalongkorn University, Bangkok, Thailand
    </p>
    <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
        © 2025 Thai AccidentIQ AI | Powered by XGBoost + SHAP + Llama 3.3 70B<br>
        Data Source: Thai Department of Highways (2019-2022)
    </p>
</div>
""", unsafe_allow_html=True)