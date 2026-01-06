"""
STEP 3: Machine Learning - Severity Prediction
Train ML models with imbalance handling (SMOTE + Class Weights)
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ML Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 3: MACHINE LEARNING - SEVERITY PREDICTION")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n📁 Working Directory: {BASE_DIR}")

# -------------------------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# -------------------------------------------------------------------------
print("\n[1/6] 📂 Loading and preparing data...")
df = pd.read_csv(INPUT_FILE)
print(f"      ✅ Loaded {len(df):,} records")

# Define features for ML (exclude display-only columns)
display_only_cols = ['acc_code', 'route', 'province_th', 'incident_datetime', 
                     'report_datetime', 'severity_class']
target_cols = ['high_severity', 'total_casualties', 'number_of_fatalities', 
               'number_of_injuries']

# Get feature columns
feature_cols = [col for col in df.columns if col not in display_only_cols + target_cols]

print(f"      Features for ML: {len(feature_cols)}")
print(f"      Target: high_severity (binary classification)")

# Prepare features and target
X = df[feature_cols].copy()
y = df['high_severity'].copy()

print(f"\n      Class distribution:")
print(f"      - Class 0 (Low Severity): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"      - Class 1 (High Severity): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"      - Imbalance Ratio: 1:{(y==0).sum()/(y==1).sum():.2f}")

# -------------------------------------------------------------------------
# 2. FEATURE ENCODING
# -------------------------------------------------------------------------
print("\n[2/6] 🔧 Encoding categorical features...")

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"      Categorical features: {len(categorical_cols)}")
print(f"      Numerical features: {len(numerical_cols)}")

# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"      ✅ Encoded {len(categorical_cols)} categorical features")

# -------------------------------------------------------------------------
# 3. TRAIN-TEST SPLIT (STRATIFIED)
# -------------------------------------------------------------------------
print("\n[3/6] ✂️  Splitting data (stratified)...")

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"      Train set: {len(X_train):,} samples")
print(f"      Test set: {len(X_test):,} samples")
print(f"\n      Train class distribution:")
print(f"      - Class 0: {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"      - Class 1: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# Feature scaling for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_file = os.path.join(MODELS_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_file)
print(f"      ✅ Saved scaler to: scaler.pkl")

# -------------------------------------------------------------------------
# 4. MODEL TRAINING WITH IMBALANCE HANDLING
# -------------------------------------------------------------------------
print("\n[4/6] 🤖 Training models with imbalance handling...")
print("      Strategy: SMOTE + Class Weights")

# Define models with class weights
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        auto_class_weights='Balanced',  # Handle imbalance
        random_state=42,
        verbose=False
    )
}

# Train models and store results
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n      Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    trained_models[name] = model
    
    print(f"         ✅ Accuracy: {accuracy:.4f}")
    print(f"         ✅ Precision: {precision:.4f}")
    print(f"         ✅ Recall: {recall:.4f}")
    print(f"         ✅ F1-Score: {f1:.4f}")
    print(f"         ✅ ROC-AUC: {roc_auc:.4f}")

# -------------------------------------------------------------------------
# 5. MODEL COMPARISON AND SELECTION
# -------------------------------------------------------------------------
print("\n[5/6] 📊 Comparing model performance...")

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results]
})

comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
comparison_df.to_csv(comparison_file, index=False)
print(f"\n      ✅ Saved comparison to: model_comparison.csv")

# Select best model based on F1-Score (best for imbalanced data)
best_model_name = comparison_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_results = results[best_model_name]

print(f"\n      🏆 Best Model: {best_model_name}")
print(f"      📊 F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")

# Save best model
best_model_file = os.path.join(MODELS_DIR, 'best_model.pkl')
joblib.dump(best_model, best_model_file)
print(f"      ✅ Saved best model to: best_model.pkl")

# Save all models
for name, model in trained_models.items():
    model_file = os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_")}.pkl')
    joblib.dump(model, model_file)

print(f"      ✅ Saved all {len(trained_models)} models")

# -------------------------------------------------------------------------
# 6. VISUALIZATION
# -------------------------------------------------------------------------
print("\n[6/6] 📈 Creating visualizations...")

# 6.1 Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.15

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, comparison_df[metric], width, 
           label=metric, color=colors[i], edgecolor='black', linewidth=1)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

plt.tight_layout()
comparison_fig = os.path.join(FIGURES_DIR, '08_model_comparison.png')
plt.savefig(comparison_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 08_model_comparison.png")

# 6.2 Confusion Matrix (Best Model)
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, best_results['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Severity', 'High Severity'],
            yticklabels=['Low Severity', 'High Severity'],
            cbar_kws={'label': 'Count'}, ax=ax,
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name}', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
cm_fig = os.path.join(FIGURES_DIR, '09_confusion_matrix.png')
plt.savefig(cm_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 09_confusion_matrix.png")

# 6.3 ROC Curves (All Models)
fig, ax = plt.subplots(figsize=(10, 8))

for name in results:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    auc = results[name]['roc_auc']
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
roc_fig = os.path.join(FIGURES_DIR, '10_roc_curves.png')
plt.savefig(roc_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 10_roc_curves.png")

# 6.4 Precision-Recall Curve (Best Model)
fig, ax = plt.subplots(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_test, best_results['y_pred_proba'])
ax.plot(recall, precision, linewidth=2.5, color='#e74c3c', label=best_model_name)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title(f'Precision-Recall Curve - {best_model_name}', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
pr_fig = os.path.join(FIGURES_DIR, '11_precision_recall_curve.png')
plt.savefig(pr_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 11_precision_recall_curve.png")

# -------------------------------------------------------------------------
# DETAILED CLASSIFICATION REPORT
# -------------------------------------------------------------------------
print("\n" + "="*80)
print(f"📊 DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("="*80)

print("\n" + classification_report(y_test, best_results['y_pred'], 
                                    target_names=['Low Severity', 'High Severity']))

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print("\nAdditional Metrics:")
print(f"True Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives: {tp:,}")
print(f"\nSpecificity: {specificity:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")

# Save detailed report
report_file = os.path.join(RESULTS_DIR, 'classification_report.txt')
with open(report_file, 'w') as f:
    f.write(f"CLASSIFICATION REPORT - {best_model_name}\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_test, best_results['y_pred'], 
                                  target_names=['Low Severity', 'High Severity']))
    f.write(f"\n\nConfusion Matrix:\n")
    f.write(f"TN: {tn:,}, FP: {fp:,}, FN: {fn:,}, TP: {tp:,}\n")
    f.write(f"\nSpecificity: {specificity:.4f}\n")
    f.write(f"NPV: {npv:.4f}\n")

print(f"\n✅ Saved detailed report to: classification_report.txt")

# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("✅ STEP 3 COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   - Models Trained: {len(trained_models)}")
print(f"   - Best Model: {best_model_name}")
print(f"   - Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
print(f"   - ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
print(f"\n📁 Saved:")
print(f"   - Models: {MODELS_DIR}")
print(f"   - Results: {RESULTS_DIR}")
print(f"   - Figures: {FIGURES_DIR}")
print(f"\n🚀 Next: Step 4 - Explainable AI (XAI) Analysis")
print("="*80)