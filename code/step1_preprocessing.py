"""
STEP 1: Data Preprocessing
Clean and prepare Thai road accident data
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np

print("="*80)
print("STEP 1: DATA PREPROCESSING")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "thai_road_accident_2019_2022.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

print(f"\n📁 Working Directory: {BASE_DIR}")

# -------------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------------
print("\n[1/6] 📂 Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"      ✅ Loaded {len(df):,} records with {len(df.columns)} columns")

# -------------------------------------------------------------------------
# 2. CONVERT DATETIME
# -------------------------------------------------------------------------
print("\n[2/6] ⏰ Converting datetime columns...")
df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
df['report_datetime'] = pd.to_datetime(df['report_datetime'])
print(f"      ✅ Date range: {df['incident_datetime'].min()} to {df['incident_datetime'].max()}")

# -------------------------------------------------------------------------
# 3. HANDLE MISSING COORDINATES
# -------------------------------------------------------------------------
print("\n[3/6] 🗺️  Handling missing coordinates...")
missing_before = df[['latitude', 'longitude']].isnull().any(axis=1).sum()
print(f"      Missing: {missing_before} records")

if missing_before > 0:
    # Fill with province median
    province_coords = df.groupby('province_en')[['latitude', 'longitude']].median()
    
    for province in df[df['latitude'].isnull()]['province_en'].unique():
        if province in province_coords.index:
            mask = (df['province_en'] == province) & (df['latitude'].isnull())
            df.loc[mask, 'latitude'] = province_coords.loc[province, 'latitude']
            df.loc[mask, 'longitude'] = province_coords.loc[province, 'longitude']
    
    # Remove any remaining missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"      ✅ Filled missing coordinates with province averages")

# -------------------------------------------------------------------------
# 4. CREATE TEMPORAL FEATURES
# -------------------------------------------------------------------------
print("\n[4/6] 🔧 Creating temporal features...")

# Date/time components
df['year'] = df['incident_datetime'].dt.year
df['month'] = df['incident_datetime'].dt.month
df['day'] = df['incident_datetime'].dt.day
df['hour'] = df['incident_datetime'].dt.hour
df['day_of_week'] = df['incident_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Thai seasons
def get_season(month):
    if month in [3, 4, 5]:
        return 'hot'
    elif month in [6, 7, 8, 9, 10]:
        return 'rainy'
    else:
        return 'cool'

df['season'] = df['month'].apply(get_season)

# Time of day
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

df['time_of_day'] = df['hour'].apply(get_time_of_day)

# Rush hour
df['is_rush_hour'] = (
    ((df['hour'] >= 7) & (df['hour'] <= 9)) | 
    ((df['hour'] >= 17) & (df['hour'] <= 19))
).astype(int)

print(f"      ✅ Created 10 temporal features")

# -------------------------------------------------------------------------
# 5. CREATE TARGET VARIABLES
# -------------------------------------------------------------------------
print("\n[5/6] 🎯 Creating target variables...")

# Total casualties
df['total_casualties'] = df['number_of_fatalities'] + df['number_of_injuries']

# Binary severity (high = any fatality OR more than 2 injuries)
df['high_severity'] = (
    (df['number_of_fatalities'] > 0) | 
    (df['number_of_injuries'] > 2)
).astype(int)

# Multi-class severity
def get_severity_class(row):
    if row['number_of_fatalities'] > 0:
        return 'fatal'
    elif row['number_of_injuries'] > 2:
        return 'major_injury'
    elif row['number_of_injuries'] > 0:
        return 'minor_injury'
    else:
        return 'no_injury'

df['severity_class'] = df.apply(get_severity_class, axis=1)

print(f"      ✅ Created 3 target variables")
print(f"\n      Target Distribution:")
print(f"      - High Severity: {df['high_severity'].sum():,} ({df['high_severity'].sum()/len(df)*100:.1f}%)")
print(f"      - Fatal: {(df['severity_class']=='fatal').sum():,}")
print(f"      - Major Injury: {(df['severity_class']=='major_injury').sum():,}")
print(f"      - Minor Injury: {(df['severity_class']=='minor_injury').sum():,}")
print(f"      - No Injury: {(df['severity_class']=='no_injury').sum():,}")

# -------------------------------------------------------------------------
# 6. SAVE PROCESSED DATA
# -------------------------------------------------------------------------
print("\n[6/6] 💾 Saving processed data...")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save full dataset
output_file = os.path.join(OUTPUT_DIR, 'preprocessed_data.csv')
df.to_csv(output_file, index=False)
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"      ✅ Saved: preprocessed_data.csv ({file_size_mb:.2f} MB)")

# Save ML columns list
display_only_cols = ['acc_code', 'route', 'province_th', 'incident_datetime', 'report_datetime']
ml_cols = [col for col in df.columns if col not in display_only_cols]

ml_cols_file = os.path.join(OUTPUT_DIR, 'ml_columns.txt')
with open(ml_cols_file, 'w', encoding='utf-8') as f:
    f.write("ML Modeling Columns\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total: {len(ml_cols)} columns\n\n")
    for i, col in enumerate(ml_cols, 1):
        f.write(f"{i:2d}. {col}\n")
    f.write(f"\n\nExcluded (Display Only): {display_only_cols}\n")

print(f"      ✅ Saved: ml_columns.txt ({len(ml_cols)} columns for ML)")

# Save sample
sample_file = os.path.join(OUTPUT_DIR, 'sample_data.csv')
df.head(100).to_csv(sample_file, index=False)
print(f"      ✅ Saved: sample_data.csv (100 rows preview)")

# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("✅ STEP 1 COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   - Total records: {len(df):,}")
print(f"   - Total columns: {len(df.columns)}")
print(f"   - ML columns: {len(ml_cols)}")
print(f"   - Date range: {df['year'].min()} - {df['year'].max()}")
print(f"\n📁 Files saved to: {OUTPUT_DIR}")
print(f"\n🚀 Next: Step 2 - Exploratory Data Analysis (EDA)")
print("="*80)