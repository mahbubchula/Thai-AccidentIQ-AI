"""
STEP 4: Explainable AI (XAI) - SHAP Analysis
Interpret model predictions and understand feature importance
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("STEP 4: EXPLAINABLE AI (XAI) - SHAP ANALYSIS")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

print(f"\n📁 Working Directory: {BASE_DIR}")

# -------------------------------------------------------------------------
# 1. LOAD DATA AND MODEL
# -------------------------------------------------------------------------
print("\n[1/7] 📂 Loading data and model...")

# Load data
df = pd.read_csv(INPUT_FILE)

# Define features
display_only_cols = ['acc_code', 'route', 'province_th', 'incident_datetime', 
                     'report_datetime', 'severity_class']
target_cols = ['high_severity', 'total_casualties', 'number_of_fatalities', 
               'number_of_injuries']
feature_cols = [col for col in df.columns if col not in display_only_cols + target_cols]

X = df[feature_cols].copy()
y = df['high_severity'].copy()

print(f"      ✅ Loaded {len(df):,} records")
print(f"      ✅ Features: {len(feature_cols)}")

# Store original feature names and categorical columns
original_feature_names = feature_cols.copy()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Load best tuned model
model_file = os.path.join(MODELS_DIR, 'best_tuned_model.pkl')
model = joblib.load(model_file)
print(f"      ✅ Loaded best tuned model: XGBoost")

# -------------------------------------------------------------------------
# 2. SHAP EXPLAINER SETUP
# -------------------------------------------------------------------------
print("\n[2/7] 🔧 Setting up SHAP explainer...")
print("      This may take a few minutes...")

# Use a sample for SHAP computation (for speed)
sample_size = 1000
np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled_df), size=sample_size, replace=False)
X_sample = X_scaled_df.iloc[sample_indices]

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print(f"      ✅ SHAP values computed for {sample_size} samples")

# -------------------------------------------------------------------------
# 3. GLOBAL FEATURE IMPORTANCE
# -------------------------------------------------------------------------
print("\n[3/7] 📊 Analyzing global feature importance...")

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

print("\n      Top 10 Most Important Features:")
for i, row in feature_importance_df.head(10).iterrows():
    print(f"      {row['feature']:.<40} {row['importance']:.4f}")

# Save feature importance
importance_file = os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')
feature_importance_df.to_csv(importance_file, index=False)
print(f"\n      ✅ Saved feature importance to: shap_feature_importance.csv")

# 3.1 Feature Importance Bar Plot
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance_df.head(15)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))

bars = ax.barh(range(len(top_features)), top_features['importance'], 
               color=colors, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=10)
ax.set_xlabel('Mean |SHAP Value| (Average Impact on Model Output)', 
              fontsize=11, fontweight='bold')
ax.set_title('Top 15 Most Important Features (SHAP)', 
             fontsize=13, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    ax.text(val + 0.001, i, f'{val:.4f}', 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
importance_fig = os.path.join(FIGURES_DIR, '14_shap_feature_importance.png')
plt.savefig(importance_fig, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 14_shap_feature_importance.png")

# -------------------------------------------------------------------------
# 4. SHAP SUMMARY PLOT
# -------------------------------------------------------------------------
print("\n[4/7] 🎨 Creating SHAP summary plot...")

# Summary plot (beeswarm)
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_sample, 
                  feature_names=feature_cols,
                  show=False, max_display=20)
plt.title('SHAP Summary Plot - Feature Impact on Predictions', 
          fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
summary_fig = os.path.join(FIGURES_DIR, '15_shap_summary_plot.png')
plt.savefig(summary_fig, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 15_shap_summary_plot.png")

# -------------------------------------------------------------------------
# 5. SHAP DEPENDENCE PLOTS (Top 6 Features)
# -------------------------------------------------------------------------
print("\n[5/7] 📈 Creating SHAP dependence plots...")

top_6_features = feature_importance_df.head(6)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, feature in enumerate(top_6_features):
    feature_idx = feature_cols.index(feature)
    
    ax = axes[idx]
    shap.dependence_plot(feature_idx, shap_values, X_sample,
                         feature_names=feature_cols,
                         show=False, ax=ax)
    ax.set_title(f'Dependence Plot: {feature}', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel(feature, fontsize=10, fontweight='bold')
    ax.set_ylabel('SHAP Value', fontsize=10, fontweight='bold')

plt.tight_layout()
dependence_fig = os.path.join(FIGURES_DIR, '16_shap_dependence_plots.png')
plt.savefig(dependence_fig, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 16_shap_dependence_plots.png")

# -------------------------------------------------------------------------
# 6. SHAP WATERFALL PLOTS (Example Predictions)
# -------------------------------------------------------------------------
print("\n[6/7] 💧 Creating SHAP waterfall plots...")

# Select examples: high-severity and low-severity predictions
high_severity_idx = np.where(y.iloc[sample_indices] == 1)[0]
low_severity_idx = np.where(y.iloc[sample_indices] == 0)[0]

# Get one example from each
high_example_idx = high_severity_idx[0] if len(high_severity_idx) > 0 else 0
low_example_idx = low_severity_idx[0] if len(low_severity_idx) > 0 else 1

examples = [
    (high_example_idx, "High Severity Accident"),
    (low_example_idx, "Low Severity Accident")
]

for idx, (example_idx, title) in enumerate(examples):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[example_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[example_idx].values,
        feature_names=feature_cols
    )
    
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title(f'SHAP Waterfall Plot - {title}', 
             fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    waterfall_fig = os.path.join(FIGURES_DIR, 
                                  f'17_shap_waterfall_{idx+1}_{title.lower().replace(" ", "_")}.png')
    plt.savefig(waterfall_fig, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"      ✅ Saved: 17_shap_waterfall_{idx+1}_{title.lower().replace(' ', '_')}.png")

# -------------------------------------------------------------------------
# 7. FEATURE INTERACTION ANALYSIS
# -------------------------------------------------------------------------
print("\n[7/7] 🔗 Analyzing feature interactions...")

# Get top 5 features
top_5_features = feature_importance_df.head(5)['feature'].tolist()

# Compute interaction values for top features
print("      Computing SHAP interaction values (this may take a moment)...")
shap_interaction_values = explainer.shap_interaction_values(X_sample)

# Create interaction heatmap for top 5 features
top_5_indices = [feature_cols.index(f) for f in top_5_features]
interaction_matrix = np.abs(shap_interaction_values[:, top_5_indices][:, :, top_5_indices]).mean(0)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(interaction_matrix, 
            xticklabels=top_5_features,
            yticklabels=top_5_features,
            annot=True, fmt='.4f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=1,
            cbar_kws={'label': 'Mean |SHAP Interaction Value|'},
            ax=ax, annot_kws={'fontsize': 9, 'fontweight': 'bold'})
ax.set_title('SHAP Feature Interaction Matrix (Top 5 Features)', 
            fontsize=13, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
interaction_fig = os.path.join(FIGURES_DIR, '18_shap_interaction_matrix.png')
plt.savefig(interaction_fig, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 18_shap_interaction_matrix.png")

# -------------------------------------------------------------------------
# INTERPRETATION SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("📊 SHAP ANALYSIS SUMMARY")
print("="*80)

print("\n🔍 Key Findings:")
print("\n1. TOP 5 MOST IMPORTANT FEATURES:")
for i, row in feature_importance_df.head(5).iterrows():
    print(f"   {i+1}. {row['feature']:.<35} Impact: {row['importance']:.4f}")

print("\n2. FEATURE INTERPRETATION:")
top_feature = feature_importance_df.iloc[0]['feature']
print(f"   • '{top_feature}' has the strongest influence on predictions")
print(f"   • This suggests {top_feature} is a critical factor in accident severity")

print("\n3. MODEL INSIGHTS:")
print("   • SHAP values show both positive and negative contributions")
print("   • Feature interactions reveal complex relationships")
print("   • Waterfall plots explain individual predictions")

# Save interpretation summary
summary_file = os.path.join(RESULTS_DIR, 'xai_interpretation_summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("EXPLAINABLE AI (XAI) INTERPRETATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("TOP 10 MOST IMPORTANT FEATURES (SHAP):\n")
    f.write("-"*80 + "\n")
    for i, row in feature_importance_df.head(10).iterrows():
        f.write(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.6f}\n")
    
    f.write("\n\nKEY INSIGHTS:\n")
    f.write("-"*80 + "\n")
    f.write(f"1. Most important feature: {feature_importance_df.iloc[0]['feature']}\n")
    f.write(f"2. Total features analyzed: {len(feature_cols)}\n")
    f.write(f"3. Samples used for SHAP: {sample_size}\n")
    f.write(f"4. Model type: XGBoost (Tuned)\n")
    
    f.write("\n\nINTERPRETATION:\n")
    f.write("-"*80 + "\n")
    f.write("• SHAP values quantify each feature's contribution to predictions\n")
    f.write("• Positive SHAP = increases probability of high severity\n")
    f.write("• Negative SHAP = decreases probability of high severity\n")
    f.write("• Feature interactions show how features work together\n")

print(f"\n✅ Saved interpretation summary to: xai_interpretation_summary.txt")

# -------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("✅ STEP 4 COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   - SHAP Analysis Complete")
print(f"   - Samples Analyzed: {sample_size:,}")
print(f"   - Features Explained: {len(feature_cols)}")
print(f"   - Figures Created: 5")
print(f"\n📁 Saved:")
print(f"   - Feature Importance: shap_feature_importance.csv")
print(f"   - Interpretation: xai_interpretation_summary.txt")
print(f"   - Figures: 14-18 in outputs/figures/")
print(f"\n📈 Figures Generated:")
print(f"   - 14_shap_feature_importance.png - Global importance ranking")
print(f"   - 15_shap_summary_plot.png - Feature impact distribution")
print(f"   - 16_shap_dependence_plots.png - Top 6 feature effects")
print(f"   - 17_shap_waterfall_1/2.png - Individual predictions explained")
print(f"   - 18_shap_interaction_matrix.png - Feature interactions")
print(f"\n🚀 Next: Step 5 - LLM Integration (Groq)")
print("="*80)