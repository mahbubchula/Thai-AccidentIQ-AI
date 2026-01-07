"""
STEP 3C: Comprehensive Spatial Analysis
Advanced geographic analysis of Thai road accidents
Author: MAHBUB Hassan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Spatial analysis libraries
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go

print("="*80)
print("STEP 3C: COMPREHENSIVE SPATIAL ANALYSIS")
print("="*80)

# Paths
BASE_DIR = r"E:\ML Research\Thai accident data"
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n📁 Working Directory: {BASE_DIR}")

# -------------------------------------------------------------------------
# 1. LOAD AND PREPARE SPATIAL DATA
# -------------------------------------------------------------------------
print("\n[1/8] 📂 Loading spatial data...")
df = pd.read_csv(INPUT_FILE)
print(f"      ✅ Loaded {len(df):,} records")

# Filter out invalid coordinates
df_spatial = df[(df['latitude'].notna()) & (df['longitude'].notna())].copy()
df_spatial = df_spatial[(df_spatial['latitude'] >= 5.0) & (df_spatial['latitude'] <= 21.0)]
df_spatial = df_spatial[(df_spatial['longitude'] >= 97.0) & (df_spatial['longitude'] <= 106.0)]

print(f"      ✅ Valid coordinates: {len(df_spatial):,} ({len(df_spatial)/len(df)*100:.1f}%)")

# Add severity flag
df_spatial['severity_numeric'] = df_spatial['high_severity']

# -------------------------------------------------------------------------
# 2. GEOGRAPHIC HOTSPOT ANALYSIS (Getis-Ord Gi*)
# -------------------------------------------------------------------------
print("\n[2/8] 🔥 Performing hotspot analysis (Getis-Ord Gi*)...")

def calculate_getis_ord(data, distance_threshold=0.5):
    """
    Calculate Getis-Ord Gi* statistic for hotspot analysis
    """
    coords = data[['latitude', 'longitude']].values
    severity = data['severity_numeric'].values
    
    # Calculate distance matrix
    distances = cdist(coords, coords, metric='euclidean')
    
    # Create spatial weights (inverse distance)
    weights = np.where(distances < distance_threshold, 
                       1 / (distances + 1e-10), 0)
    np.fill_diagonal(weights, 0)
    
    # Normalize weights
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, where=row_sums!=0)
    
    # Calculate Gi* statistic
    n = len(severity)
    mean_severity = severity.mean()
    std_severity = severity.std()
    
    gi_stats = []
    z_scores = []
    
    for i in range(n):
        if row_sums[i] > 0:
            local_sum = (weights[i] * severity).sum()
            expected = mean_severity * weights[i].sum()
            
            # Z-score
            variance = std_severity**2 * weights[i].sum()
            if variance > 0:
                z = (local_sum - expected) / np.sqrt(variance)
            else:
                z = 0
        else:
            z = 0
            
        gi_stats.append(local_sum)
        z_scores.append(z)
    
    return np.array(gi_stats), np.array(z_scores)

# Sample data for computation (use all if possible, sample for speed)
if len(df_spatial) > 10000:
    print(f"      Sampling 10,000 points for computation speed...")
    sample_spatial = df_spatial.sample(n=10000, random_state=42)
else:
    sample_spatial = df_spatial.copy()

gi_stats, z_scores = calculate_getis_ord(sample_spatial, distance_threshold=0.3)
sample_spatial['gi_star'] = gi_stats
sample_spatial['z_score'] = z_scores

# Classify hotspots
sample_spatial['hotspot_type'] = 'Not Significant'
sample_spatial.loc[z_scores > 1.96, 'hotspot_type'] = 'Hot Spot (High Risk)'
sample_spatial.loc[z_scores < -1.96, 'hotspot_type'] = 'Cold Spot (Low Risk)'

print(f"      ✅ Hot Spots (z > 1.96): {(z_scores > 1.96).sum()} locations")
print(f"      ✅ Cold Spots (z < -1.96): {(z_scores < -1.96).sum()} locations")

# Save hotspot results
hotspot_summary = sample_spatial.groupby('hotspot_type').agg({
    'latitude': 'count',
    'severity_numeric': 'mean'
}).round(4)
hotspot_summary.columns = ['Count', 'Avg_Severity_Rate']
hotspot_file = os.path.join(RESULTS_DIR, 'spatial_hotspot_analysis.csv')
hotspot_summary.to_csv(hotspot_file)
print(f"      ✅ Saved: spatial_hotspot_analysis.csv")

# -------------------------------------------------------------------------
# 3. GLOBAL SPATIAL AUTOCORRELATION (Moran's I)
# -------------------------------------------------------------------------
print("\n[3/8] 📊 Calculating Global Moran's I...")

def calculate_morans_i(data, distance_threshold=0.5):
    """
    Calculate Global Moran's I for spatial autocorrelation
    """
    coords = data[['latitude', 'longitude']].values
    severity = data['severity_numeric'].values
    
    # Distance matrix
    distances = cdist(coords, coords, metric='euclidean')
    
    # Spatial weights (binary: 1 if within threshold, 0 otherwise)
    weights = (distances < distance_threshold).astype(int)
    np.fill_diagonal(weights, 0)
    
    # Calculate Moran's I
    n = len(severity)
    W = weights.sum()
    
    if W == 0:
        return 0, 0, 0
    
    y = severity - severity.mean()
    
    # Moran's I formula
    numerator = n * np.sum(weights * np.outer(y, y))
    denominator = W * np.sum(y**2)
    
    I = numerator / denominator if denominator != 0 else 0
    
    # Expected value under null hypothesis
    E_I = -1 / (n - 1)
    
    # Variance (simplified)
    var_I = (n / W) * (1 / (n - 1))
    
    # Z-score
    z = (I - E_I) / np.sqrt(var_I) if var_I > 0 else 0
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return I, z, p_value

morans_i, z_moran, p_moran = calculate_morans_i(sample_spatial, distance_threshold=0.3)

print(f"      Moran's I: {morans_i:.4f}")
print(f"      Z-score: {z_moran:.4f}")
print(f"      P-value: {p_moran:.4f}")

if p_moran < 0.05:
    if morans_i > 0:
        interpretation = "CLUSTERED (accidents are spatially concentrated)"
    else:
        interpretation = "DISPERSED (accidents are spatially spread out)"
else:
    interpretation = "RANDOM (no significant spatial pattern)"

print(f"      Interpretation: {interpretation}")

# Save Moran's I results
morans_results = pd.DataFrame({
    'Statistic': ['Morans_I', 'Z_Score', 'P_Value', 'Interpretation'],
    'Value': [morans_i, z_moran, p_moran, interpretation]
})
morans_file = os.path.join(RESULTS_DIR, 'spatial_morans_i.csv')
morans_results.to_csv(morans_file, index=False)
print(f"      ✅ Saved: spatial_morans_i.csv")

# -------------------------------------------------------------------------
# 4. PROVINCE-LEVEL RISK ANALYSIS
# -------------------------------------------------------------------------
print("\n[4/8] 🗺️  Analyzing province-level risk patterns...")

province_stats = df_spatial.groupby('province_en').agg({
    'latitude': 'count',
    'severity_numeric': ['sum', 'mean'],
    'number_of_fatalities': 'sum',
    'number_of_injuries': 'sum'
}).round(4)

province_stats.columns = ['Total_Accidents', 'High_Severity_Count', 
                          'Severity_Rate', 'Total_Fatalities', 'Total_Injuries']

# Calculate risk score
province_stats['Risk_Score'] = (
    province_stats['Severity_Rate'] * 0.5 + 
    (province_stats['Total_Fatalities'] / province_stats['Total_Accidents']) * 0.3 +
    (province_stats['Total_Injuries'] / province_stats['Total_Accidents']) * 0.2
)

province_stats = province_stats.sort_values('Risk_Score', ascending=False)

print(f"\n      Top 10 Highest Risk Provinces:")
for i, (prov, row) in enumerate(province_stats.head(10).iterrows(), 1):
    print(f"      {i:2d}. {prov:.<35} Risk Score: {row['Risk_Score']:.4f}")

# Save province analysis
province_file = os.path.join(RESULTS_DIR, 'spatial_province_risk.csv')
province_stats.to_csv(province_file)
print(f"\n      ✅ Saved: spatial_province_risk.csv")

# -------------------------------------------------------------------------
# 5. KERNEL DENSITY ESTIMATION
# -------------------------------------------------------------------------
print("\n[5/8] 🌡️  Performing Kernel Density Estimation...")

from scipy.stats import gaussian_kde

# Create grid for KDE
lat_min, lat_max = df_spatial['latitude'].min(), df_spatial['latitude'].max()
lon_min, lon_max = df_spatial['longitude'].min(), df_spatial['longitude'].max()

# Grid resolution
resolution = 100
lat_grid = np.linspace(lat_min, lat_max, resolution)
lon_grid = np.linspace(lon_min, lon_max, resolution)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# High severity accidents only
high_severity = df_spatial[df_spatial['high_severity'] == 1]

if len(high_severity) > 100:
    # Sample if too many points
    if len(high_severity) > 5000:
        high_severity_sample = high_severity.sample(n=5000, random_state=42)
    else:
        high_severity_sample = high_severity
    
    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
    values = np.vstack([high_severity_sample['longitude'], high_severity_sample['latitude']])
    
    kernel = gaussian_kde(values)
    density = np.reshape(kernel(positions).T, lon_mesh.shape)
    
    print(f"      ✅ KDE computed for {len(high_severity_sample):,} high-severity accidents")
else:
    print(f"      ⚠️  Too few high-severity accidents for KDE")
    density = None

# -------------------------------------------------------------------------
# 6. URBAN VS RURAL ANALYSIS
# -------------------------------------------------------------------------
print("\n[6/8] 🏙️  Analyzing Urban vs Rural patterns...")

# Define major urban provinces
urban_provinces = ['Bangkok', 'Nonthaburi', 'Samut Prakan', 'Pathum Thani', 
                   'Chiang Mai', 'Phuket', 'Chonburi']

df_spatial['location_type'] = df_spatial['province_en'].apply(
    lambda x: 'Urban' if x in urban_provinces else 'Rural'
)

urban_rural_stats = df_spatial.groupby('location_type').agg({
    'latitude': 'count',
    'severity_numeric': 'mean',
    'number_of_fatalities': 'sum',
    'number_of_injuries': 'sum'
}).round(4)

urban_rural_stats.columns = ['Total_Accidents', 'Severity_Rate', 
                              'Total_Fatalities', 'Total_Injuries']

print(f"\n      Urban vs Rural Comparison:")
print(urban_rural_stats.to_string())

# Statistical test
urban_severity = df_spatial[df_spatial['location_type'] == 'Urban']['severity_numeric']
rural_severity = df_spatial[df_spatial['location_type'] == 'Rural']['severity_numeric']
t_stat, p_value = stats.ttest_ind(urban_severity, rural_severity)

print(f"\n      T-test: t={t_stat:.4f}, p={p_value:.4f}")
print(f"      Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

# Save urban/rural analysis
urban_rural_file = os.path.join(RESULTS_DIR, 'spatial_urban_rural.csv')
urban_rural_stats.to_csv(urban_rural_file)
print(f"      ✅ Saved: spatial_urban_rural.csv")

# -------------------------------------------------------------------------
# 7. SPATIAL CLUSTERING (DBSCAN)
# -------------------------------------------------------------------------
print("\n[7/8] 🎯 Detecting spatial clusters (DBSCAN)...")

# High severity accidents only
high_sev_coords = df_spatial[df_spatial['high_severity'] == 1][['latitude', 'longitude']].values

if len(high_sev_coords) > 100:
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.1, min_samples=10, metric='euclidean')
    clusters = dbscan.fit_predict(high_sev_coords)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"      ✅ Clusters found: {n_clusters}")
    print(f"      ✅ Noise points: {n_noise}")
    
    # Add cluster labels
    df_spatial_high = df_spatial[df_spatial['high_severity'] == 1].copy()
    df_spatial_high['cluster'] = clusters
    
    # Cluster statistics
    cluster_stats = df_spatial_high[df_spatial_high['cluster'] != -1].groupby('cluster').agg({
        'latitude': ['count', 'mean'],
        'longitude': 'mean',
        'number_of_fatalities': 'sum'
    })
    
    print(f"\n      Top 5 Largest Clusters:")
    for i, (cluster_id, row) in enumerate(cluster_stats.nlargest(5, ('latitude', 'count')).iterrows(), 1):
        print(f"      {i}. Cluster {cluster_id}: {row[('latitude', 'count')]} accidents at ({row[('latitude', 'mean')]:.2f}, {row[('longitude', 'mean')]:.2f})")
else:
    print(f"      ⚠️  Too few points for clustering")
    clusters = None

# -------------------------------------------------------------------------
# 8. VISUALIZATIONS
# -------------------------------------------------------------------------
print("\n[8/8] 📈 Creating spatial visualizations...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# 8.1 Hotspot Map
fig, ax = plt.subplots(figsize=(14, 10))
scatter = ax.scatter(sample_spatial['longitude'], sample_spatial['latitude'],
                    c=sample_spatial['z_score'], s=20, cmap='RdYlBu_r',
                    alpha=0.6, edgecolors='black', linewidth=0.5,
                    vmin=-3, vmax=3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Getis-Ord Gi* Z-Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax.set_title('Spatial Hotspot Analysis (Getis-Ord Gi*)', 
            fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
hotspot_fig = os.path.join(FIGURES_DIR, '19_spatial_hotspot_map.png')
plt.savefig(hotspot_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 19_spatial_hotspot_map.png")

# 8.2 Province Risk Map
fig, ax = plt.subplots(figsize=(12, 10))
top_20 = province_stats.head(20)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_20)))
bars = ax.barh(range(len(top_20)), top_20['Risk_Score'], color=colors,
              edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20.index, fontsize=10)
ax.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Provinces by Risk Score', 
            fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_20['Risk_Score'])):
    ax.text(val + 0.01, i, f'{val:.3f}', 
           va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
province_fig = os.path.join(FIGURES_DIR, '20_spatial_province_risk.png')
plt.savefig(province_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 20_spatial_province_risk.png")

# 8.3 KDE Heat Map
if density is not None:
    fig, ax = plt.subplots(figsize=(14, 10))
    contour = ax.contourf(lon_mesh, lat_mesh, density, levels=20, cmap='YlOrRd', alpha=0.7)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Accident Density', fontsize=12, fontweight='bold')
    
    # Overlay high severity points
    ax.scatter(high_severity_sample['longitude'], high_severity_sample['latitude'],
              s=1, c='black', alpha=0.3, label='High Severity Accidents')
    
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Kernel Density Estimation - High Severity Accidents', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    kde_fig = os.path.join(FIGURES_DIR, '21_spatial_kde_heatmap.png')
    plt.savefig(kde_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved: 21_spatial_kde_heatmap.png")

# 8.4 Urban vs Rural Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Severity rate
urban_rural_stats['Severity_Rate'].plot(kind='bar', ax=axes[0], 
                                        color=['#e74c3c', '#3498db'],
                                        edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Location Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Severity Rate', fontsize=12, fontweight='bold')
axes[0].set_title('Severity Rate: Urban vs Rural', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(True, alpha=0.3, axis='y')

# Accident count
urban_rural_stats['Total_Accidents'].plot(kind='bar', ax=axes[1],
                                          color=['#e74c3c', '#3498db'],
                                          edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Location Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Total Accidents', fontsize=12, fontweight='bold')
axes[1].set_title('Accident Count: Urban vs Rural', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
urban_rural_fig = os.path.join(FIGURES_DIR, '22_spatial_urban_rural.png')
plt.savefig(urban_rural_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✅ Saved: 22_spatial_urban_rural.png")

# 8.5 Moran's I Scatterplot
if len(sample_spatial) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate spatial lag (simplified)
    coords = sample_spatial[['latitude', 'longitude']].values
    severity = sample_spatial['severity_numeric'].values
    
    distances = cdist(coords, coords, metric='euclidean')
    weights = (distances < 0.3).astype(int)
    np.fill_diagonal(weights, 0)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights_norm = np.divide(weights, row_sums, where=row_sums!=0)
    
    spatial_lag = (weights_norm @ severity.reshape(-1, 1)).flatten()
    
    # Standardize
    severity_std = (severity - severity.mean()) / severity.std()
    lag_std = (spatial_lag - spatial_lag.mean()) / spatial_lag.std()
    
    ax.scatter(severity_std, lag_std, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    m, b = np.polyfit(severity_std[~np.isnan(severity_std) & ~np.isnan(lag_std)], 
                      lag_std[~np.isnan(severity_std) & ~np.isnan(lag_std)], 1)
    ax.plot(severity_std, m*severity_std + b, 'r-', linewidth=2, 
           label=f"Moran's I = {morans_i:.4f}")
    
    # Add quadrant lines
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Standardized Severity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spatial Lag (Standardized)', fontsize=12, fontweight='bold')
    ax.set_title("Moran's I Scatterplot", fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    moran_fig = os.path.join(FIGURES_DIR, '23_spatial_morans_scatterplot.png')
    plt.savefig(moran_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved: 23_spatial_morans_scatterplot.png")

# -------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("📊 SPATIAL ANALYSIS SUMMARY")
print("="*80)

print(f"\n🔍 KEY FINDINGS:")
print(f"\n1. HOTSPOT ANALYSIS:")
print(f"   - Hot Spots: {(z_scores > 1.96).sum()} high-risk locations")
print(f"   - Cold Spots: {(z_scores < -1.96).sum()} low-risk locations")

print(f"\n2. SPATIAL AUTOCORRELATION:")
print(f"   - Moran's I: {morans_i:.4f}")
print(f"   - Pattern: {interpretation}")
print(f"   - Significance: {'YES (p < 0.05)' if p_moran < 0.05 else 'NO (p ≥ 0.05)'}")

print(f"\n3. PROVINCE RISK:")
top3 = province_stats.head(3)
for i, (prov, row) in enumerate(top3.iterrows(), 1):
    print(f"   {i}. {prov}: Risk Score = {row['Risk_Score']:.4f}")

print(f"\n4. URBAN VS RURAL:")
urban_rate = urban_rural_stats.loc['Urban', 'Severity_Rate']
rural_rate = urban_rural_stats.loc['Rural', 'Severity_Rate']
print(f"   - Urban Severity Rate: {urban_rate:.4f}")
print(f"   - Rural Severity Rate: {rural_rate:.4f}")
print(f"   - Difference: {'Urban higher' if urban_rate > rural_rate else 'Rural higher'}")

if clusters is not None:
    print(f"\n5. SPATIAL CLUSTERS:")
    print(f"   - Number of clusters: {n_clusters}")
    print(f"   - Clustered accidents: {len(clusters) - n_noise}")

print(f"\n" + "="*80)
print("✅ STEP 3C COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   - Spatial analyses: 6 completed")
print(f"   - Figures created: 5")
print(f"   - Statistical tests: Moran's I, T-test, Getis-Ord Gi*")
print(f"\n📁 Saved:")
print(f"   - Results: {RESULTS_DIR}")
print(f"   - Figures: {FIGURES_DIR}")
print(f"\n🎓 For Paper:")
print(f"   - Use figures 19-23 for spatial analysis section")
print(f"   - Report Moran's I, hotspot analysis, province risk scores")
print(f"   - Discuss urban-rural differences")
print("="*80)