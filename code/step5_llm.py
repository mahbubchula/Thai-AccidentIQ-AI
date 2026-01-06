"""
STEP 5: LLM Integration - Intelligent Insights
Use Groq/Llama for automated analysis and recommendations
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np
from groq import Groq
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 5: LLM INTEGRATION - INTELLIGENT INSIGHTS")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"\n📁 Working Directory: {BASE_DIR}")

# -------------------------------------------------------------------------
# 1. SETUP GROQ CLIENT
# -------------------------------------------------------------------------
print("\n[1/5] 🔧 Setting up Groq LLM client...")

# Your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    client = Groq(api_key=GROQ_API_KEY)
    print(f"      ✅ Groq client initialized")
    print(f"      Model: llama-3.3-70b-versatile")
except Exception as e:
    print(f"      ❌ Error: {e}")
    print(f"      Please check your API key")
    exit(1)

# -------------------------------------------------------------------------
# 2. LOAD RESEARCH RESULTS
# -------------------------------------------------------------------------
print("\n[2/5] 📂 Loading research results...")

# Load feature importance
importance_file = os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')
feature_importance = pd.read_csv(importance_file)

# Load model comparison
comparison_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
model_comparison = pd.read_csv(comparison_file)

# Load tuning results
tuning_file = os.path.join(RESULTS_DIR, 'tuning_comparison.csv')
tuning_comparison = pd.read_csv(tuning_file)

print(f"      ✅ Loaded research results")

# -------------------------------------------------------------------------
# 3. GENERATE AUTOMATED INSIGHTS
# -------------------------------------------------------------------------
print("\n[3/5] 🤖 Generating LLM-powered insights...")

def query_llm(prompt, system_message="You are an expert transportation safety analyst and data scientist."):
    """Query Groq LLM"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# 3.1 Interpret Feature Importance
print("\n      [1/4] Interpreting feature importance...")

top_features = feature_importance.head(10)
features_text = "\n".join([f"{i+1}. {row['feature']}: {row['importance']:.4f}" 
                           for i, row in top_features.iterrows()])

prompt_features = f"""
Analyze these SHAP feature importance results from a road accident severity prediction model in Thailand:

TOP 10 FEATURES:
{features_text}

Provide:
1. Key insights about what drives accident severity
2. Surprising or notable patterns
3. Policy implications (3-4 specific recommendations)
4. Why these features might be important in Thai context

Keep it concise and actionable. Focus on practical implications.
"""

insights_features = query_llm(prompt_features)
print(f"      ✅ Feature importance insights generated")

# 3.2 Interpret Model Performance
print("\n      [2/4] Analyzing model performance...")

best_model = model_comparison.iloc[0]
prompt_performance = f"""
Analyze this machine learning model performance for predicting road accident severity:

BEST MODEL: {best_model['Model']}
- Accuracy: {best_model['Accuracy']:.4f}
- Precision: {best_model['Precision']:.4f}
- Recall: {best_model['Recall']:.4f}
- F1-Score: {best_model['F1-Score']:.4f}
- ROC-AUC: {best_model['ROC-AUC']:.4f}

Context: 
- This is an imbalanced dataset (82.3% low severity, 17.7% high severity)
- Goal is to catch high-severity accidents for emergency response

Provide:
1. Interpretation of these metrics in practical terms
2. What the recall score means for real-world deployment
3. Strengths and limitations of the model
4. Recommendations for implementation

Be specific about what these numbers mean for saving lives.
"""

insights_performance = query_llm(prompt_performance)
print(f"      ✅ Performance insights generated")

# 3.3 Safety Recommendations
print("\n      [3/4] Generating safety recommendations...")

prompt_safety = f"""
Based on this Thai road accident research findings:

KEY FINDINGS:
- Vehicle type is the #1 predictor of severity
- Accident type is #2
- Multi-vehicle accidents are more severe
- Geographic location matters significantly
- Time factors are less important than expected

Dataset: 81,735 accidents in Thailand (2019-2022)
- 10,135 fatal accidents (12.4%)
- Main cause: Speeding (74%)
- Most common: Pickup trucks (35%)

Generate 5 specific, evidence-based policy recommendations for Thai Department of Highways. 
Each should include:
- The recommendation
- Expected impact
- Implementation difficulty (Low/Medium/High)
- Target timeline

Be practical and Thailand-specific.
"""

safety_recommendations = query_llm(prompt_safety)
print(f"      ✅ Safety recommendations generated")

# 3.4 Executive Summary
print("\n      [4/4] Creating executive summary...")

prompt_summary = f"""
Create an executive summary for transportation authorities based on:

RESEARCH: Machine Learning Analysis of Thai Road Accidents (2019-2022)
- Dataset: 81,735 accidents
- Best Model: XGBoost with F1-Score: 0.52, ROC-AUC: 0.81
- Key Finding: Vehicle type is strongest predictor

Top 3 Risk Factors:
1. Vehicle type
2. Accident type (rollover vs collision)
3. Number of vehicles involved

Create a 200-word executive summary suitable for:
- Ministry of Transport officials
- Policy makers
- Non-technical audience

Include: main findings, implications, and top recommendation.
"""

executive_summary = query_llm(prompt_summary)
print(f"      ✅ Executive summary generated")

# -------------------------------------------------------------------------
# 4. INTERACTIVE Q&A SYSTEM
# -------------------------------------------------------------------------
print("\n[4/5] 💬 Testing interactive Q&A system...")

# Example questions
example_questions = [
    "What time of day has the highest severity rate?",
    "Which provinces should be prioritized for safety interventions?",
    "How can we reduce motorcycle accident severity?"
]

qa_responses = {}

for i, question in enumerate(example_questions, 1):
    print(f"\n      Q{i}: {question}")
    
    context = f"""
    You are analyzing Thai road accident data (2019-2022, 81,735 accidents).
    
    Key facts:
    - Top risk factor: Vehicle type
    - 74% caused by speeding
    - 12.4% are fatal
    - Most common: Pickup trucks (35%), Motorcycles (18%)
    
    Question: {question}
    
    Provide a brief, data-driven answer (2-3 sentences).
    """
    
    answer = query_llm(context)
    qa_responses[question] = answer
    print(f"      A: {answer[:100]}...")

print(f"\n      ✅ Q&A system tested")

# -------------------------------------------------------------------------
# 5. SAVE ALL REPORTS
# -------------------------------------------------------------------------
print("\n[5/5] 💾 Saving LLM-generated reports...")

# 5.1 Comprehensive Report
full_report = f"""
================================================================================
THAI ROAD ACCIDENT ANALYSIS - LLM-POWERED INSIGHTS REPORT
================================================================================
Generated using Groq Llama 3.1 70B

DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
EXECUTIVE SUMMARY
================================================================================

{executive_summary}

================================================================================
1. FEATURE IMPORTANCE ANALYSIS
================================================================================

{insights_features}

================================================================================
2. MODEL PERFORMANCE INTERPRETATION
================================================================================

{insights_performance}

================================================================================
3. SAFETY RECOMMENDATIONS
================================================================================

{safety_recommendations}

================================================================================
4. EXAMPLE Q&A RESPONSES
================================================================================

"""

for question, answer in qa_responses.items():
    full_report += f"\nQ: {question}\nA: {answer}\n"
    full_report += "-" * 80 + "\n"

full_report += f"""
================================================================================
METHODOLOGY NOTE
================================================================================

This report was generated using:
- Machine Learning: XGBoost (F1-Score: 0.52, ROC-AUC: 0.81)
- Explainability: SHAP analysis
- Language Model: Groq Llama 3.3 70B Versatile
- Dataset: 81,735 Thai road accidents (2019-2022)

The insights combine quantitative ML predictions with qualitative LLM interpretation
to provide actionable recommendations for transportation safety policy.

================================================================================
"""

# Save full report
report_file = os.path.join(REPORTS_DIR, 'llm_insights_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(full_report)

print(f"      ✅ Saved: llm_insights_report.txt")

# 5.2 Executive Summary Only
exec_file = os.path.join(REPORTS_DIR, 'executive_summary.txt')
with open(exec_file, 'w', encoding='utf-8') as f:
    f.write("EXECUTIVE SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(executive_summary)
    f.write("\n\n" + "="*80 + "\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"      ✅ Saved: executive_summary.txt")

# 5.3 Policy Recommendations
policy_file = os.path.join(REPORTS_DIR, 'policy_recommendations.txt')
with open(policy_file, 'w', encoding='utf-8') as f:
    f.write("POLICY RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    f.write(safety_recommendations)
    f.write("\n\n" + "="*80 + "\n")

print(f"      ✅ Saved: policy_recommendations.txt")

# 5.4 JSON Output (for web deployment)
json_output = {
    "executive_summary": executive_summary,
    "feature_insights": insights_features,
    "model_performance": insights_performance,
    "safety_recommendations": safety_recommendations,
    "qa_examples": qa_responses,
    "metadata": {
        "generated_at": pd.Timestamp.now().isoformat(),
        "model": "llama-3.3-70b-versatile",
        "dataset_size": 81735,
        "best_ml_model": "XGBoost"
    }
}

json_file = os.path.join(REPORTS_DIR, 'llm_insights.json')
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, indent=2, ensure_ascii=False)

print(f"      ✅ Saved: llm_insights.json")

# -------------------------------------------------------------------------
# DEMONSTRATION
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("📊 LLM INSIGHTS PREVIEW")
print("="*80)

print("\n🎯 EXECUTIVE SUMMARY:")
print("-" * 80)
print(executive_summary[:300] + "...\n")

print("\n🔍 TOP FEATURE INSIGHT:")
print("-" * 80)
print(insights_features[:300] + "...\n")

print("\n🚦 SAMPLE RECOMMENDATION:")
print("-" * 80)
print(safety_recommendations[:300] + "...\n")

# -------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("✅ STEP 5 COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   - LLM Model: Llama 3.3 70B Versatile (Groq)")
print(f"   - Reports Generated: 4")
print(f"   - Insights Created: Feature Analysis, Performance, Safety, Q&A")
print(f"\n📁 Saved:")
print(f"   - Full Report: llm_insights_report.txt")
print(f"   - Executive Summary: executive_summary.txt")
print(f"   - Policy Recommendations: policy_recommendations.txt")
print(f"   - JSON Data: llm_insights.json")
print(f"\n💡 Key Capabilities Demonstrated:")
print(f"   ✅ Automated insight generation from ML results")
print(f"   ✅ Natural language interpretation of SHAP analysis")
print(f"   ✅ Evidence-based policy recommendations")
print(f"   ✅ Interactive Q&A system")
print(f"   ✅ Executive summary generation")
print(f"\n🚀 Next: Step 6 - Streamlit Deployment")
print("="*80)