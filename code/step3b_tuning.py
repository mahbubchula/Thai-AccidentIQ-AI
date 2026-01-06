"""
STEP 3B: Hyperparameter Tuning + Cross-Validation
Optimize ML models for best performance with rigorous evaluation
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, make_scorer

# ML Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Hyperparameter tuning
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

import joblib
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*80)
print("STEP 3B: HYPERPARAMETER TUNING + CROSS-VALIDATION")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

print(f"\n📁 Working Directory: {BASE_DIR}")

# -------------------------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# -------------------------------------------------------------------------
print("\n[1/5] 📂 Loading and preparing data...")
df = pd.read_csv(INPUT_FILE)
print(f"      ✅ Loaded {len(df):,} records")

# Define features for ML
display_only_cols = ['acc_code', 'route', 'province_th', 'incident_datetime', 
                     'report_datetime', 'severity_class']
target_cols = ['high_severity', 'total_casualties', 'number_of_fatalities', 
               'number_of_injuries']

feature_cols = [col for col in df.columns if col not in display_only_cols + target_cols]

X = df[feature_cols].copy()
y = df['high_severity'].copy()

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"      ✅ Features: {X.shape[1]}")
print(f"      ✅ Samples: {X.shape[0]:,}")
print(f"      ✅ Imbalance Ratio: 1:{(y==0).sum()/(y==1).sum():.2f}")

# -------------------------------------------------------------------------
# 2. BASELINE CROSS-VALIDATION (Default Parameters)
# -------------------------------------------------------------------------
print("\n[2/5] 📊 Baseline Cross-Validation (5-Fold)...")
print("      Testing default parameters from Step 3...")

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score)

# Baseline models (same as Step 3)
baseline_models = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y==0).sum()/(y==1).sum(),
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    )
}

baseline_results = {}

for name, model in baseline_models.items():
    print(f"\n      Evaluating {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
    
    baseline_results[name] = {
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    print(f"         F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"         Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

print("\n      ✅ Baseline cross-validation complete")

# -------------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING WITH OPTUNA
# -------------------------------------------------------------------------
print("\n[3/5] 🔧 Hyperparameter Tuning (Optuna)...")
print("      This may take 10-15 minutes...")

tuned_models = {}
tuning_results = {}

# 3.1 XGBoost Tuning
print("\n      [1/3] Tuning XGBoost...")

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': (y==0).sum()/(y==1).sum(),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
    return cv_scores.mean()

study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print(f"         ✅ Best F1-Score: {study_xgb.best_value:.4f}")
print(f"         Best parameters: {study_xgb.best_params}")

tuned_models['XGBoost'] = XGBClassifier(**study_xgb.best_params, 
                                         scale_pos_weight=(y==0).sum()/(y==1).sum(),
                                         random_state=42, n_jobs=-1, eval_metric='logloss')
tuning_results['XGBoost'] = {
    'best_f1': study_xgb.best_value,
    'best_params': study_xgb.best_params,
    'study': study_xgb
}

# 3.2 LightGBM Tuning
print("\n      [2/3] Tuning LightGBM...")

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
    return cv_scores.mean()

study_lgb = optuna.create_study(direction='maximize', study_name='LightGBM')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

print(f"         ✅ Best F1-Score: {study_lgb.best_value:.4f}")
print(f"         Best parameters: {study_lgb.best_params}")

tuned_models['LightGBM'] = LGBMClassifier(**study_lgb.best_params, 
                                           class_weight='balanced',
                                           random_state=42, n_jobs=-1, verbose=-1)
tuning_results['LightGBM'] = {
    'best_f1': study_lgb.best_value,
    'best_params': study_lgb.best_params,
    'study': study_lgb
}

# 3.3 CatBoost Tuning
print("\n      [3/3] Tuning CatBoost...")

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 300),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'auto_class_weights': 'Balanced',
        'random_state': 42,
        'verbose': False
    }
    
    model = CatBoostClassifier(**params)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
    return cv_scores.mean()

study_cat = optuna.create_study(direction='maximize', study_name='CatBoost')
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

print(f"         ✅ Best F1-Score: {study_cat.best_value:.4f}")
print(f"         Best parameters: {study_cat.best_params}")

tuned_models['CatBoost'] = CatBoostClassifier(**study_cat.best_params,
                                               auto_class_weights='Balanced',
                                               random_state=42, verbose=False)
tuning_results['CatBoost'] = {
    'best_f1': study_cat.best_value,
    'best_params': study_cat.best_params,
    'study': study_cat
}

print("\n      ✅ Hyperparameter tuning complete!")

# -------------------------------------------------------------------------
# 4. FINAL EVALUATION WITH TUNED MODELS
# -------------------------------------------------------------------------
print("\n[4/5] 🏆 Final Evaluation (Tuned Models)...")

tuned_cv_results = {}

for name, model in tuned_models.items():
    print(f"\n      Evaluating Tuned {name}...")
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
    
    tuned_cv_results[name] = {
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    print(f"         F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"         Improvement: {(cv_scores.mean() - baseline_results[name]['mean_f1']):.4f}")

# -------------------------------------------------------------------------
# 5. COMPARISON AND SAVING
# -------------------------------------------------------------------------
print("\n[5/5] 📊 Creating comparison and saving results...")

# Create comparison dataframe
comparison_data = []

for name in baseline_models.keys():
    comparison_data.append({
        'Model': name,
        'Type': 'Baseline',
        'Mean F1': baseline_results[name]['mean_f1'],
        'Std F1': baseline_results[name]['std_f1'],
        'Parameters': 'Default'
    })
    
    comparison_data.append({
        'Model': name,
        'Type': 'Tuned',
        'Mean F1': tuned_cv_results[name]['mean_f1'],
        'Std F1': tuned_cv_results[name]['std_f1'],
        'Parameters': 'Optimized'
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*80)
print("📊 BASELINE vs TUNED COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_file = os.path.join(RESULTS_DIR, 'tuning_comparison.csv')
comparison_df.to_csv(comparison_file, index=False)
print(f"\n✅ Saved comparison to: tuning_comparison.csv")

# Find best overall model
best_tuned = max(tuned_cv_results.items(), key=lambda x: x[1]['mean_f1'])
best_model_name = best_tuned[0]
best_f1 = best_tuned[1]['mean_f1']

print(f"\n🏆 Best Tuned Model: {best_model_name}")
print(f"📊 Cross-Validated F1-Score: {best_f1:.4f} (±{best_tuned[1]['std_f1']:.4f})")

# Train final model on full dataset
print(f"\n🔧 Training final {best_model_name} on full dataset...")
final_model = tuned_models[best_model_name]
final_model.fit(X_scaled, y)

# Save final tuned model
final_model_file = os.path.join(MODELS_DIR, 'best_tuned_model.pkl')
joblib.dump(final_model, final_model_file)
print(f"✅ Saved final tuned model to: best_tuned_model.pkl")

# Save tuned parameters
params_file = os.path.join(RESULTS_DIR, 'best_tuned_parameters.txt')
with open(params_file, 'w', encoding='utf-8') as f:
    f.write(f"BEST TUNED MODEL: {best_model_name}\n")
    f.write("="*80 + "\n\n")
    f.write(f"Cross-Validated F1-Score: {best_f1:.4f} (±{best_tuned[1]['std_f1']:.4f})\n\n")
    f.write("Best Hyperparameters:\n")
    for param, value in tuning_results[best_model_name]['best_params'].items():
        f.write(f"  {param}: {value}\n")

print(f"✅ Saved parameters to: best_tuned_parameters.txt")

# -------------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------------
print("\n📈 Creating visualizations...")

# 5.1 Baseline vs Tuned Comparison
fig, ax = plt.subplots(figsize=(12, 6))

models_list = list(baseline_models.keys())
x = np.arange(len(models_list))
width = 0.35

baseline_means = [baseline_results[m]['mean_f1'] for m in models_list]
baseline_stds = [baseline_results[m]['std_f1'] for m in models_list]
tuned_means = [tuned_cv_results[m]['mean_f1'] for m in models_list]
tuned_stds = [tuned_cv_results[m]['std_f1'] for m in models_list]

bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
               label='Baseline', color='#3498db', edgecolor='black', 
               linewidth=1.5, capsize=5)
bars2 = ax.bar(x + width/2, tuned_means, width, yerr=tuned_stds,
               label='Tuned', color='#2ecc71', edgecolor='black', 
               linewidth=1.5, capsize=5)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score (5-Fold CV)', fontsize=12, fontweight='bold')
ax.set_title('Baseline vs Tuned Models Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(tuned_means) * 1.15)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
comparison_fig = os.path.join(FIGURES_DIR, '12_tuning_comparison.png')
plt.savefig(comparison_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 12_tuning_comparison.png")

# 5.2 Improvement Chart
fig, ax = plt.subplots(figsize=(10, 6))

improvements = [(tuned_cv_results[m]['mean_f1'] - baseline_results[m]['mean_f1']) * 100 
                for m in models_list]
colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

bars = ax.barh(models_list, improvements, color=colors, edgecolor='black', linewidth=1.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('F1-Score Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Improvement After Tuning', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, imp) in enumerate(zip(bars, improvements)):
    label = f'+{imp:.2f}%' if imp > 0 else f'{imp:.2f}%'
    x_pos = imp + (0.1 if imp > 0 else -0.1)
    ax.text(x_pos, i, label, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
improvement_fig = os.path.join(FIGURES_DIR, '13_improvement_chart.png')
plt.savefig(improvement_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 13_improvement_chart.png")

# -------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("✅ STEP 3B COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   - Cross-Validation: 5-Fold Stratified")
print(f"   - Tuning Trials per Model: 50")
print(f"   - Best Model: {best_model_name}")
print(f"   - Best F1-Score: {best_f1:.4f} (±{best_tuned[1]['std_f1']:.4f})")
print(f"   - Improvement: {(best_f1 - baseline_results[best_model_name]['mean_f1']):.4f}")
print(f"\n📁 Saved:")
print(f"   - Final Model: best_tuned_model.pkl")
print(f"   - Parameters: best_tuned_parameters.txt")
print(f"   - Comparison: tuning_comparison.csv")
print(f"   - Figures: 12_tuning_comparison.png, 13_improvement_chart.png")
print(f"\n🚀 Next: Step 4 - Explainable AI (XAI) with SHAP")
print("="*80)
