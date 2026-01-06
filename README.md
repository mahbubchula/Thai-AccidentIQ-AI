# 🚗 Thai AccidentIQ AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)](https://xgboost.ai)

> **Intelligent Road Safety Analytics & Prediction System for Thailand**

Advanced machine learning system for predicting and analyzing road accident severity in Thailand using 81,735 historical accidents (2019-2022).

🌐 **Live Demo**: [Coming Soon]

---

## 🎯 Overview

Thai AccidentIQ AI combines **Machine Learning**, **Explainable AI (XAI)**, and **Large Language Models** to provide:

- 🔮 **Real-time accident severity prediction** (F1-Score: 0.523, ROC-AUC: 0.81)
- 📊 **Interactive analytics dashboard** with 18 publication-quality visualizations
- 🧠 **SHAP-based explainability** for interpretable predictions
- 💬 **AI-powered chat assistant** for safety insights
- 📄 **Automated policy recommendations** using LLM

---

## ✨ Key Features

### 🤖 Machine Learning
- **4 Optimized Models**: XGBoost, LightGBM, CatBoost, Random Forest
- **Hyperparameter Tuning**: 50 trials per model using Optuna
- **5-Fold Cross-Validation**: Stratified sampling for robust evaluation
- **Imbalance Handling**: Class weights + stratified splits

### 🔍 Explainable AI (XAI)
- **SHAP Analysis**: Global and local feature importance
- **Dependence Plots**: Feature effect visualization
- **Waterfall Plots**: Individual prediction explanations
- **Interaction Analysis**: Feature relationship discovery

### 🤖 LLM Integration
- **Groq Llama 3.3 70B**: Natural language insights
- **Automated Reports**: Executive summaries and policy recommendations
- **Interactive Chat**: Real-time Q&A system

### 🌐 Web Application
- **Modern UI**: AI-themed design with glassmorphism
- **Real-time Predictions**: Interactive severity forecasting
- **Live Analytics**: Dynamic data visualization
- **Report Downloads**: CSV, TXT, JSON exports

---

## 📊 Research Results

| Metric | Value | Details |
|--------|-------|---------|
| **Dataset** | 81,735 accidents | Thai roads (2019-2022) |
| **Best Model** | XGBoost | Hyperparameter optimized |
| **F1-Score** | 0.523 | 5-fold CV mean |
| **ROC-AUC** | 0.81 | Excellent discriminative ability |
| **Recall** | 67.5% | Catches 2/3 of high-severity cases |
| **Top Predictor** | Vehicle Type | SHAP importance: 0.47 |

### 🔍 Key Findings

1. **Vehicle type** is the strongest predictor of severity (SHAP: 0.47)
2. **Speeding** causes 74% of all accidents
3. **Geographic location** significantly impacts outcomes
4. **Multi-vehicle accidents** are 3× more severe
5. **Temporal factors** have less impact than expected

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mahbubchula/Thai-AccidentIQ-AI.git
cd Thai-AccidentIQ-AI
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

5. **Download dataset**

Place `thai_road_accident_2019_2022.csv` in `data/raw/`

### Usage

#### Run Complete Analysis Pipeline

```bash
# Step 1: Data Preprocessing
python code/step1_preprocessing.py

# Step 2: Exploratory Data Analysis
python code/step2_eda.py

# Step 3: ML Modeling
python code/step3_ml.py

# Step 3B: Hyperparameter Tuning
python code/step3b_tuning.py

# Step 4: XAI Analysis
python code/step4_xai.py

# Step 5: LLM Integration
python code/step5_llm.py
```

#### Launch Web Application

```bash
streamlit run thai_accidentiq_ai.py
```

Access at: `http://localhost:8501`

---

## 📁 Project Structure

```
Thai-AccidentIQ-AI/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned data
│   └── external/               # Additional data
├── code/
│   ├── step1_preprocessing.py  # Data cleaning
│   ├── step2_eda.py           # Exploratory analysis
│   ├── step3_ml.py            # ML modeling
│   ├── step3b_tuning.py       # Hyperparameter optimization
│   ├── step4_xai.py           # SHAP analysis
│   └── step5_llm.py           # LLM integration
├── models/
│   ├── best_tuned_model.pkl   # Optimized XGBoost
│   ├── scaler.pkl             # Feature scaler
│   └── *.pkl                  # Other models
├── outputs/
│   ├── figures/               # 18 publication figures
│   ├── reports/               # Research reports
│   └── results/               # Performance metrics
├── thai_accidentiq_ai.py      # Streamlit app
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git exclusions
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 🔬 Methodology

### Data Processing
1. **Cleaning**: Handle missing values, remove duplicates
2. **Feature Engineering**: Temporal, geospatial, categorical
3. **Target Creation**: Binary, multi-class, regression targets

### Machine Learning
1. **Models**: XGBoost, LightGBM, CatBoost, Random Forest
2. **Optimization**: Bayesian optimization (Optuna, 50 trials)
3. **Validation**: 5-fold stratified cross-validation
4. **Metrics**: F1-Score, ROC-AUC, Precision, Recall

### Explainability
1. **SHAP**: Feature importance and interactions
2. **LIME**: Local explanations
3. **Visualizations**: 6 publication-quality figures

### LLM Integration
1. **Model**: Groq Llama 3.3 70B Versatile
2. **Tasks**: Insight generation, Q&A, recommendations
3. **Outputs**: Executive summaries, policy briefs

---

## 📊 Results & Visualizations

### Sample Outputs

**Feature Importance (SHAP)**
- Top predictor: Vehicle Type (0.47)
- Second: Accident Type (0.34)
- Third: Number of Vehicles (0.32)

**Model Performance**
- XGBoost: F1=0.523, AUC=0.81
- LightGBM: F1=0.519, AUC=0.81
- CatBoost: F1=0.509, AUC=0.79

**Policy Recommendations**
1. Vehicle-specific safety interventions
2. Geographic targeting of resources
3. Multi-vehicle accident prevention
4. Speed enforcement in high-risk zones

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@software{hassan2025thai,
  author = {Hassan, MAHBUB},
  title = {Thai AccidentIQ AI: Intelligent Road Safety Analytics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mahbubchula/Thai-AccidentIQ-AI},
  institution = {Chulalongkorn University},
  department = {Department of Civil Engineering}
}
```

---

## 👨‍🎓 Author

**MAHBUB Hassan**

Graduate Student & Non ASEAN Scholar  
Department of Civil Engineering  
Faculty of Engineering  
Chulalongkorn University  
Bangkok, Thailand

📧 Email: 6870376421@student.chula.ac.th  
🔗 GitHub: [@mahbubchula](https://github.com/mahbubchula)

**Research Interests:**
- Transportation Safety
- Machine Learning Applications
- Climate-Adaptive Systems
- AI in Transportation Engineering

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Source**: Office of the Permanent Secretary, Ministry of Transport, Thailand
- **Institution**: Chulalongkorn University, Bangkok
- **Funding**: Non ASEAN Scholarship Program
- **Technologies**: XGBoost, SHAP, Groq, Streamlit

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/mahbubchula/Thai-AccidentIQ-AI/issues)
- **Email**: 6870376421@student.chula.ac.th
- **University**: Chulalongkorn University

---

## ⚠️ Disclaimer

This research is for academic purposes. Policy recommendations should be validated by transportation authorities before implementation. The model's predictions are probabilistic and should be used as decision support, not as the sole basis for critical decisions.

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**© 2025 MAHBUB Hassan | Chulalongkorn University | All Rights Reserved**
