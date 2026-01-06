"""
STEP 2: Exploratory Data Analysis (EDA)
Publication-quality visualizations for Thai road accident data
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

print("="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📁 Working Directory: {BASE_DIR}")
print(f"📊 Figures will be saved to: {OUTPUT_DIR}")

# -------------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------------
print("\n[1/8] 📂 Loading processed data...")
df = pd.read_csv(INPUT_FILE)
df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
print(f"      ✅ Loaded {len(df):,} records with {len(df.columns)} columns")

# -------------------------------------------------------------------------
# 2. CLASS IMBALANCE ANALYSIS
# -------------------------------------------------------------------------
print("\n[2/8] ⚖️  Analyzing class imbalance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary classification
severity_counts = df['high_severity'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Low Severity', 'High Severity'], 
            [severity_counts[0], severity_counts[1]], 
            color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Number of Accidents', fontsize=12, fontweight='bold')
axes[0].set_title('Binary Classification - Class Imbalance', 
                  fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylim(0, severity_counts.max() * 1.1)

# Add percentage labels
for i, (label, count) in enumerate(zip(['Low Severity', 'High Severity'], 
                                        [severity_counts[0], severity_counts[1]])):
    pct = count / len(df) * 100
    axes[0].text(i, count + 1000, f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Multi-class classification
severity_class_counts = df['severity_class'].value_counts()
colors_multi = ['#3498db', '#f39c12', '#e74c3c', '#9b59b6']
severity_order = ['no_injury', 'minor_injury', 'major_injury', 'fatal']
counts_ordered = [severity_class_counts[s] for s in severity_order]
labels_ordered = ['No Injury', 'Minor Injury', 'Major Injury', 'Fatal']

axes[1].bar(labels_ordered, counts_ordered, 
            color=colors_multi, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Number of Accidents', fontsize=12, fontweight='bold')
axes[1].set_title('Multi-class Classification - Severity Distribution', 
                  fontsize=14, fontweight='bold', pad=20)
axes[1].tick_params(axis='x', rotation=15)
axes[1].set_ylim(0, max(counts_ordered) * 1.1)

# Add percentage labels
for i, (label, count) in enumerate(zip(labels_ordered, counts_ordered)):
    pct = count / len(df) * 100
    axes[1].text(i, count + 500, f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
imbalance_file = os.path.join(OUTPUT_DIR, '01_class_imbalance.png')
plt.savefig(imbalance_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 01_class_imbalance.png")

# -------------------------------------------------------------------------
# 3. TEMPORAL TRENDS
# -------------------------------------------------------------------------
print("\n[3/8] 📅 Analyzing temporal trends...")

# Accidents by year
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# By Year
yearly = df.groupby('year').size()
axes[0, 0].bar(yearly.index, yearly.values, color='#3498db', 
               edgecolor='black', linewidth=1.5)
axes[0, 0].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Accidents by Year', fontsize=13, fontweight='bold', pad=15)
for i, v in enumerate(yearly.values):
    axes[0, 0].text(yearly.index[i], v + 200, f'{v:,}', 
                   ha='center', fontsize=10, fontweight='bold')

# By Month
monthly = df.groupby('month').size()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0, 1].plot(monthly.index, monthly.values, marker='o', 
                linewidth=2.5, markersize=8, color='#e74c3c')
axes[0, 1].fill_between(monthly.index, monthly.values, alpha=0.3, color='#e74c3c')
axes[0, 1].set_xlabel('Month', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Accidents by Month', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_names, rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# By Hour
hourly = df.groupby('hour').size()
axes[1, 0].bar(hourly.index, hourly.values, color='#f39c12', 
               edgecolor='black', linewidth=1.5)
axes[1, 0].axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush')
axes[1, 0].axvspan(17, 19, alpha=0.2, color='red', label='Evening Rush')
axes[1, 0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Accidents by Hour (Rush Hours Highlighted)', 
                     fontsize=13, fontweight='bold', pad=15)
axes[1, 0].legend(loc='upper right')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# By Day of Week
dow = df.groupby('day_of_week').size()
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
             'Friday', 'Saturday', 'Sunday']
colors_dow = ['#3498db']*5 + ['#e74c3c', '#e74c3c']  # Highlight weekends
axes[1, 1].bar(range(7), dow.values, color=colors_dow, 
               edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel('Day of Week', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Accidents by Day of Week (Weekends in Red)', 
                     fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(dow_names, rotation=45, ha='right')

plt.tight_layout()
temporal_file = os.path.join(OUTPUT_DIR, '02_temporal_trends.png')
plt.savefig(temporal_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 02_temporal_trends.png")

# -------------------------------------------------------------------------
# 4. SEVERITY ANALYSIS BY TIME
# -------------------------------------------------------------------------
print("\n[4/8] 🎯 Analyzing severity patterns...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Severity by Hour
severity_by_hour = df.groupby(['hour', 'high_severity']).size().unstack(fill_value=0)
severity_pct = severity_by_hour.div(severity_by_hour.sum(axis=1), axis=0) * 100

axes[0, 0].plot(severity_pct.index, severity_pct[1], 
                marker='o', linewidth=2.5, markersize=8, color='#e74c3c', label='High Severity')
axes[0, 0].fill_between(severity_pct.index, severity_pct[1], alpha=0.3, color='#e74c3c')
axes[0, 0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('High Severity Rate (%)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('High Severity Rate by Hour', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Severity by Season
severity_by_season = df.groupby(['season', 'high_severity']).size().unstack(fill_value=0)
severity_season_pct = severity_by_season.div(severity_by_season.sum(axis=1), axis=0) * 100
season_order = ['hot', 'rainy', 'cool']
season_labels = ['Hot\n(Mar-May)', 'Rainy\n(Jun-Oct)', 'Cool\n(Nov-Feb)']

x = np.arange(len(season_order))
width = 0.35
axes[0, 1].bar(x - width/2, [severity_season_pct.loc[s, 0] for s in season_order], 
               width, label='Low Severity', color='#2ecc71', edgecolor='black')
axes[0, 1].bar(x + width/2, [severity_season_pct.loc[s, 1] for s in season_order], 
               width, label='High Severity', color='#e74c3c', edgecolor='black')
axes[0, 1].set_xlabel('Season', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Severity Distribution by Thai Season', 
                     fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(season_labels)
axes[0, 1].legend()

# Severity by Time of Day
severity_by_tod = df.groupby(['time_of_day', 'high_severity']).size().unstack(fill_value=0)
severity_tod_pct = severity_by_tod.div(severity_by_tod.sum(axis=1), axis=0) * 100
tod_order = ['morning', 'afternoon', 'evening', 'night']
tod_labels = ['Morning\n(6-12)', 'Afternoon\n(12-17)', 'Evening\n(17-21)', 'Night\n(21-6)']

x = np.arange(len(tod_order))
axes[1, 0].bar(x - width/2, [severity_tod_pct.loc[s, 0] for s in tod_order], 
               width, label='Low Severity', color='#2ecc71', edgecolor='black')
axes[1, 0].bar(x + width/2, [severity_tod_pct.loc[s, 1] for s in tod_order], 
               width, label='High Severity', color='#e74c3c', edgecolor='black')
axes[1, 0].set_xlabel('Time of Day', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Severity Distribution by Time of Day', 
                     fontsize=13, fontweight='bold', pad=15)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(tod_labels)
axes[1, 0].legend()

# Fatal accidents by vehicle type (top 10)
fatal_by_vehicle = df[df['severity_class'] == 'fatal']['vehicle_type'].value_counts().head(10)
axes[1, 1].barh(range(len(fatal_by_vehicle)), fatal_by_vehicle.values, 
                color='#e74c3c', edgecolor='black', linewidth=1.5)
axes[1, 1].set_yticks(range(len(fatal_by_vehicle)))
axes[1, 1].set_yticklabels(fatal_by_vehicle.index, fontsize=9)
axes[1, 1].set_xlabel('Number of Fatal Accidents', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Top 10 Vehicle Types in Fatal Accidents', 
                     fontsize=13, fontweight='bold', pad=15)
axes[1, 1].invert_yaxis()

for i, v in enumerate(fatal_by_vehicle.values):
    axes[1, 1].text(v + 50, i, f'{v:,}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
severity_file = os.path.join(OUTPUT_DIR, '03_severity_analysis.png')
plt.savefig(severity_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 03_severity_analysis.png")

# -------------------------------------------------------------------------
# 5. VEHICLE TYPE ANALYSIS
# -------------------------------------------------------------------------
print("\n[5/8] 🚗 Analyzing vehicle types...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 10 vehicle types
vehicle_counts = df['vehicle_type'].value_counts().head(10)
axes[0].barh(range(len(vehicle_counts)), vehicle_counts.values, 
             color='#3498db', edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(vehicle_counts)))
axes[0].set_yticklabels(vehicle_counts.index, fontsize=10)
axes[0].set_xlabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[0].set_title('Top 10 Vehicle Types in Accidents', 
                  fontsize=13, fontweight='bold', pad=15)
axes[0].invert_yaxis()

for i, v in enumerate(vehicle_counts.values):
    axes[0].text(v + 300, i, f'{v:,}', va='center', fontsize=10, fontweight='bold')

# Severity rate by vehicle type (top 10)
vehicle_severity = df.groupby('vehicle_type')['high_severity'].agg(['sum', 'count'])
vehicle_severity['rate'] = (vehicle_severity['sum'] / vehicle_severity['count'] * 100)
vehicle_severity = vehicle_severity[vehicle_severity['count'] >= 500].sort_values('rate', ascending=False).head(10)

axes[1].barh(range(len(vehicle_severity)), vehicle_severity['rate'].values, 
             color='#e74c3c', edgecolor='black', linewidth=1.5)
axes[1].set_yticks(range(len(vehicle_severity)))
axes[1].set_yticklabels(vehicle_severity.index, fontsize=10)
axes[1].set_xlabel('High Severity Rate (%)', fontsize=11, fontweight='bold')
axes[1].set_title('High Severity Rate by Vehicle Type (min 500 accidents)', 
                  fontsize=13, fontweight='bold', pad=15)
axes[1].invert_yaxis()

for i, v in enumerate(vehicle_severity['rate'].values):
    axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
vehicle_file = os.path.join(OUTPUT_DIR, '04_vehicle_analysis.png')
plt.savefig(vehicle_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 04_vehicle_analysis.png")

# -------------------------------------------------------------------------
# 6. WEATHER AND ROAD CONDITIONS
# -------------------------------------------------------------------------
print("\n[6/8] ☁️  Analyzing weather and road conditions...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Weather condition distribution
weather_counts = df['weather_condition'].value_counts()
axes[0, 0].bar(range(len(weather_counts)), weather_counts.values, 
               color='#3498db', edgecolor='black', linewidth=1.5)
axes[0, 0].set_xticks(range(len(weather_counts)))
axes[0, 0].set_xticklabels(weather_counts.index, rotation=45, ha='right', fontsize=10)
axes[0, 0].set_ylabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Accidents by Weather Condition', 
                     fontsize=13, fontweight='bold', pad=15)

for i, v in enumerate(weather_counts.values):
    axes[0, 0].text(i, v + 500, f'{v:,}', ha='center', fontsize=9, fontweight='bold')

# Severity by weather
weather_severity = df.groupby('weather_condition')['high_severity'].agg(['sum', 'count'])
weather_severity['rate'] = (weather_severity['sum'] / weather_severity['count'] * 100)
weather_severity = weather_severity.sort_values('rate', ascending=False)

axes[0, 1].bar(range(len(weather_severity)), weather_severity['rate'].values, 
               color='#e74c3c', edgecolor='black', linewidth=1.5)
axes[0, 1].set_xticks(range(len(weather_severity)))
axes[0, 1].set_xticklabels(weather_severity.index, rotation=45, ha='right', fontsize=10)
axes[0, 1].set_ylabel('High Severity Rate (%)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('High Severity Rate by Weather', 
                     fontsize=13, fontweight='bold', pad=15)

for i, v in enumerate(weather_severity['rate'].values):
    axes[0, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Road description
road_counts = df['road_description'].value_counts().head(10)
axes[1, 0].barh(range(len(road_counts)), road_counts.values, 
                color='#f39c12', edgecolor='black', linewidth=1.5)
axes[1, 0].set_yticks(range(len(road_counts)))
axes[1, 0].set_yticklabels(road_counts.index, fontsize=10)
axes[1, 0].set_xlabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Top 10 Road Types', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].invert_yaxis()

# Slope description
slope_counts = df['slope_description'].value_counts()
colors_slope = ['#2ecc71', '#f39c12', '#e74c3c'][:len(slope_counts)]
axes[1, 1].pie(slope_counts.values, labels=slope_counts.index, autopct='%1.1f%%',
               colors=colors_slope, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1, 1].set_title('Accidents by Slope Condition', 
                     fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
conditions_file = os.path.join(OUTPUT_DIR, '05_weather_road_conditions.png')
plt.savefig(conditions_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 05_weather_road_conditions.png")

# -------------------------------------------------------------------------
# 7. TOP PROVINCES
# -------------------------------------------------------------------------
print("\n[7/8] 🗺️  Analyzing provincial distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 15 provinces by accident count
province_counts = df['province_en'].value_counts().head(15)
axes[0].barh(range(len(province_counts)), province_counts.values, 
             color='#9b59b6', edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(province_counts)))
axes[0].set_yticklabels(province_counts.index, fontsize=10)
axes[0].set_xlabel('Number of Accidents', fontsize=11, fontweight='bold')
axes[0].set_title('Top 15 Provinces by Accident Count', 
                  fontsize=13, fontweight='bold', pad=15)
axes[0].invert_yaxis()

for i, v in enumerate(province_counts.values):
    axes[0].text(v + 50, i, f'{v:,}', va='center', fontsize=10, fontweight='bold')

# Top 15 provinces by fatality rate
province_fatal = df.groupby('province_en').agg({
    'number_of_fatalities': 'sum',
    'acc_code': 'count'
})
province_fatal['fatal_rate'] = (province_fatal['number_of_fatalities'] / 
                                 province_fatal['acc_code'] * 100)
province_fatal = province_fatal[province_fatal['acc_code'] >= 100].sort_values('fatal_rate', ascending=False).head(15)

axes[1].barh(range(len(province_fatal)), province_fatal['fatal_rate'].values, 
             color='#e74c3c', edgecolor='black', linewidth=1.5)
axes[1].set_yticks(range(len(province_fatal)))
axes[1].set_yticklabels(province_fatal.index, fontsize=10)
axes[1].set_xlabel('Fatality Rate (%) per Accident', fontsize=11, fontweight='bold')
axes[1].set_title('Top 15 Provinces by Fatality Rate (min 100 accidents)', 
                  fontsize=13, fontweight='bold', pad=15)
axes[1].invert_yaxis()

for i, v in enumerate(province_fatal['fatal_rate'].values):
    axes[1].text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
province_file = os.path.join(OUTPUT_DIR, '06_provincial_analysis.png')
plt.savefig(province_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 06_provincial_analysis.png")

# -------------------------------------------------------------------------
# 8. CORRELATION HEATMAP
# -------------------------------------------------------------------------
print("\n[8/8] 🔥 Creating correlation heatmap...")

# Select numerical features for correlation
numerical_cols = ['number_of_vehicles_involved', 'number_of_fatalities', 
                  'number_of_injuries', 'total_casualties', 'high_severity',
                  'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour']

corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={'fontsize': 9, 'fontweight': 'bold'})
plt.title('Correlation Matrix - Numerical Features', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

corr_file = os.path.join(OUTPUT_DIR, '07_correlation_heatmap.png')
plt.savefig(corr_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"      ✅ Saved: 07_correlation_heatmap.png")

# -------------------------------------------------------------------------
# SUMMARY STATISTICS
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("📊 EDA SUMMARY STATISTICS")
print("="*80)

print(f"\n1. CLASS IMBALANCE:")
print(f"   - Low Severity: {(df['high_severity']==0).sum():,} ({(df['high_severity']==0).sum()/len(df)*100:.1f}%)")
print(f"   - High Severity: {(df['high_severity']==1).sum():,} ({(df['high_severity']==1).sum()/len(df)*100:.1f}%)")
print(f"   - Imbalance Ratio: 1:{(df['high_severity']==0).sum()/(df['high_severity']==1).sum():.1f}")

print(f"\n2. SEVERITY BREAKDOWN:")
for sev in ['no_injury', 'minor_injury', 'major_injury', 'fatal']:
    count = (df['severity_class']==sev).sum()
    print(f"   - {sev.replace('_', ' ').title()}: {count:,} ({count/len(df)*100:.1f}%)")

print(f"\n3. TEMPORAL PATTERNS:")
print(f"   - Peak Hour: {df.groupby('hour').size().idxmax()}:00")
print(f"   - Peak Month: {month_names[df.groupby('month').size().idxmax()-1]}")
print(f"   - Peak Day: {dow_names[df.groupby('day_of_week').size().idxmax()]}")

print(f"\n4. TOP RISK FACTORS:")
print(f"   - Most Common Vehicle: {df['vehicle_type'].mode()[0]}")
print(f"   - Most Common Cause: {df['presumed_cause'].value_counts().index[0]}")
print(f"   - Most Common Accident Type: {df['accident_type'].value_counts().index[0]}")
print(f"   - Highest Risk Weather: {weather_severity.index[0]} ({weather_severity['rate'].iloc[0]:.1f}% high severity)")

print("\n" + "="*80)
print("✅ STEP 2 COMPLETE!")
print("="*80)
print(f"\n📊 Total Figures Created: 7")
print(f"📁 Saved to: {OUTPUT_DIR}")
print(f"\n🚀 Next: Step 3 - Feature Engineering & ML Modeling")
print("="*80)