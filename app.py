import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import io
import base64
from pathlib import Path
import math
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation

# Page configuration
st.set_page_config(
    page_title="TransitFair: Mapping Transit Inequity",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def calculate_distances_to_transit(gtfs_df, census_df):
    """
    Calculate distance from each census tract to nearest transit stop
    """
    distances = []
    
    for _, census_row in census_df.iterrows():
        tract_lat = census_row['latitude']
        tract_lon = census_row['longitude']
        
        min_distance = float('inf')
        nearest_stop = None
        
        for _, stop_row in gtfs_df.iterrows():
            stop_lat = stop_row['stop_lat']
            stop_lon = stop_row['stop_lon']
            
            distance = haversine_distance(tract_lat, tract_lon, stop_lat, stop_lon)
            
            if distance < min_distance:
                min_distance = distance
                nearest_stop = stop_row['stop_name']
        
        distances.append({
            'tract_id': census_row['tract_id'],
            'tract_name': census_row.get('tract_name', census_row['tract_id']),
            'latitude': tract_lat,
            'longitude': tract_lon,
            'total_population': census_row['total_population'],
            'median_income': census_row['median_income'],
            'car_ownership_rate': census_row.get('car_ownership_rate', 0.3),
            'population_density': census_row.get('population_density', 0),
            'area_sq_km': census_row.get('area_sq_km', 1.0),
            'poverty_rate': census_row.get('poverty_rate', 0.15),
            'distance_to_nearest_stop_km': min_distance,
            'nearest_stop_name': nearest_stop
        })
    
    return pd.DataFrame(distances)

def create_geodataframe(df, lat_col='latitude', lon_col='longitude'):
    """
    Convert DataFrame to GeoDataFrame with Point geometry
    """
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf

def normalize_features(distances_df):
    """
    Normalize features for machine learning using MinMaxScaler
    """
    # Select features for normalization
    feature_columns = ['median_income', 'car_ownership_rate', 'distance_to_nearest_stop_km']
    
    # Add population density if not present, calculate it
    if 'population_density' not in distances_df.columns:
        distances_df['population_density'] = distances_df['total_population'] / distances_df['area_sq_km']
    
    feature_columns.append('population_density')
    
    # Create feature matrix
    feature_matrix = distances_df[feature_columns].copy()
    
    # Handle missing values
    feature_matrix = feature_matrix.fillna(feature_matrix.mean())
    
    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Create DataFrame with normalized features
    normalized_df = pd.DataFrame(
        normalized_features,
        columns=[f'{col}_normalized' for col in feature_columns],
        index=distances_df.index
    )
    
    return normalized_df, scaler, feature_columns

def perform_kmeans_clustering(normalized_df, n_clusters=4):
    """
    Perform K-means clustering on normalized features
    """
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_df)
    
    return cluster_labels, kmeans

def analyze_clusters(distances_df, cluster_labels):
    """
    Analyze cluster characteristics and assign equity labels
    """
    # Add cluster labels to dataframe
    distances_df_with_clusters = distances_df.copy()
    distances_df_with_clusters['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = distances_df_with_clusters.groupby('cluster').agg({
        'distance_to_nearest_stop_km': ['mean', 'std'],
        'median_income': ['mean', 'std'],
        'total_population': ['mean', 'sum'],
        'car_ownership_rate': 'mean',
        'population_density': 'mean'
    }).round(3)
    
    # Flatten column names
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
    
    # Sort clusters by average distance (most underserved first)
    cluster_stats = cluster_stats.sort_values('distance_to_nearest_stop_km_mean')
    
    # Assign equity labels (extend for more clusters if needed)
    base_equity_labels = [
        "🚨 Severely Underserved",
        "⚠️ Moderately Underserved", 
        "✅ Adequately Served",
        "🌟 Well Served"
    ]
    
    # Extend labels if we have more clusters
    equity_labels = {}
    for i in range(len(cluster_stats)):
        if i < len(base_equity_labels):
            equity_labels[i] = base_equity_labels[i]
        else:
            equity_labels[i] = f"Cluster {i+1}"
    
    # Map cluster numbers to equity labels based on sorting
    cluster_to_equity = {}
    for i, cluster_num in enumerate(cluster_stats.index):
        cluster_to_equity[cluster_num] = equity_labels[i]
    
    # Add equity labels to dataframe
    distances_df_with_clusters['equity_label'] = distances_df_with_clusters['cluster'].map(cluster_to_equity)
    
    return distances_df_with_clusters, cluster_stats, cluster_to_equity

def calculate_transit_equity_score(distances_df, weights=None):
    """
    Calculate transit equity score using the formula:
    transit_equity_score = 0.4 * population_density + 0.4 * (1/distance_km) + 0.2 * (1/income)
    """
    if weights is None:
        weights = {'population_density': 0.4, 'distance': 0.4, 'income': 0.2}
    
    # Ensure population density exists
    if 'population_density' not in distances_df.columns:
        distances_df['population_density'] = distances_df['total_population'] / distances_df['area_sq_km']
    
    # Calculate components
    # Normalize population density (0-1 scale)
    pop_density_norm = (distances_df['population_density'] - distances_df['population_density'].min()) / \
                      (distances_df['population_density'].max() - distances_df['population_density'].min())
    
    # Distance component (inverse, so closer = higher score)
    distance_component = 1 / (distances_df['distance_to_nearest_stop_km'] + 0.1)  # Add small constant to avoid division by zero
    distance_norm = (distance_component - distance_component.min()) / (distance_component.max() - distance_component.min())
    
    # Income component (inverse, so lower income = higher need = higher score)
    income_component = 1 / (distances_df['median_income'] + 1000)  # Add constant to avoid division by zero
    income_norm = (income_component - income_component.min()) / (income_component.max() - income_component.min())
    
    # Calculate weighted equity score
    equity_score = (weights['population_density'] * pop_density_norm + 
                   weights['distance'] * distance_norm + 
                   weights['income'] * income_norm)
    
    # Normalize to 0-1 scale
    equity_score_norm = (equity_score - equity_score.min()) / (equity_score.max() - equity_score.min())
    
    return equity_score_norm

def create_equity_choropleth_map(distances_df, gtfs_df):
    """
    Create a choropleth map showing transit equity scores
    """
    # Calculate map center
    center_lat = (gtfs_df['stop_lat'].mean() + distances_df['latitude'].mean()) / 2
    center_lon = (gtfs_df['stop_lon'].mean() + distances_df['longitude'].mean()) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add transit stops
    for idx, row in gtfs_df.head(30).iterrows():
        folium.Marker(
            [row['stop_lat'], row['stop_lon']],
            popup=f"<b>{row['stop_name']}</b><br>ID: {row['stop_id']}",
            icon=folium.Icon(color='blue', icon='train')
        ).add_to(m)
    
    # Add census tracts with equity score coloring
    for idx, row in distances_df.head(40).iterrows():
        equity_score = row['transit_equity_score']
        
        # Color based on equity score (red=low, yellow=medium, green=high)
        if equity_score < 0.3:
            color = 'red'
        elif equity_score < 0.6:
            color = 'orange'
        elif equity_score < 0.8:
            color = 'yellow'
        else:
            color = 'green'
        
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=15,
            popup=f"<b>{row['tract_name']}</b><br>Equity Score: {equity_score:.3f}<br>Population: {row['total_population']:,}<br>Income: ${row['median_income']:,}<br>Distance: {row['distance_to_nearest_stop_km']:.2f} km",
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def propose_new_stops(distances_df, gtfs_df, equity_threshold=0.3, max_proposals=5):
    """
    Propose new transit stops for areas with low equity scores
    """
    # Filter tracts with low equity scores
    low_equity_tracts = distances_df[distances_df['transit_equity_score'] < equity_threshold].copy()
    
    if len(low_equity_tracts) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Sort by equity score (lowest first) and take top proposals
    low_equity_tracts = low_equity_tracts.nsmallest(max_proposals, 'transit_equity_score')
    
    proposed_stops = []
    impact_analysis = []
    
    for idx, tract in low_equity_tracts.iterrows():
        # Proposed stop location (tract centroid)
        proposed_lat = tract['latitude']
        proposed_lon = tract['longitude']
        
        # Calculate new distances for all tracts if this stop is added
        new_distances = []
        for _, other_tract in distances_df.iterrows():
            # Calculate distance to proposed stop
            distance_to_proposed = haversine_distance(
                other_tract['latitude'], other_tract['longitude'],
                proposed_lat, proposed_lon
            )
            
            # Take the minimum of current distance and distance to proposed stop
            current_distance = other_tract['distance_to_nearest_stop_km']
            new_distance = min(current_distance, distance_to_proposed)
            
            new_distances.append({
                'tract_id': other_tract['tract_id'],
                'tract_name': other_tract['tract_name'],
                'current_distance': current_distance,
                'new_distance': new_distance,
                'improvement': current_distance - new_distance
            })
        
        # Calculate equity score improvement
        new_distances_df = pd.DataFrame(new_distances)
        
        # Recalculate equity scores with new distances
        distances_with_new_stop = distances_df.copy()
        distances_with_new_stop['distance_to_nearest_stop_km'] = new_distances_df['new_distance']
        
        # Recalculate equity scores
        new_equity_scores = calculate_transit_equity_score(distances_with_new_stop)
        distances_with_new_stop['new_equity_score'] = new_equity_scores
        
        # Calculate improvement
        equity_improvement = new_equity_scores - distances_df['transit_equity_score']
        
        # Add proposed stop
        proposed_stops.append({
            'proposed_stop_id': f"PROPOSED_{len(proposed_stops)+1:03d}",
            'proposed_stop_name': f"New Stop - {tract['tract_name']}",
            'latitude': proposed_lat,
            'longitude': proposed_lon,
            'tract_name': tract['tract_name'],
            'current_equity_score': tract['transit_equity_score'],
            'population_served': tract['total_population'],
            'median_income': tract['median_income'],
            'current_distance': tract['distance_to_nearest_stop_km'],
            'nearest_existing_stop': tract['nearest_stop_name']
        })
        
        # Add impact analysis
        impact_analysis.append({
            'proposed_stop': f"New Stop - {tract['tract_name']}",
            'tract_name': tract['tract_name'],
            'current_equity_score': tract['transit_equity_score'],
            'new_equity_score': new_equity_scores.loc[idx],
            'equity_improvement': equity_improvement.loc[idx],
            'equity_improvement_pct': (equity_improvement.loc[idx] / tract['transit_equity_score']) * 100 if tract['transit_equity_score'] > 0 else 0,
            'avg_distance_improvement': new_distances_df['improvement'].mean(),
            'tracts_improved': len(new_distances_df[new_distances_df['improvement'] > 0]),
            'total_population_impacted': distances_df.loc[new_distances_df[new_distances_df['improvement'] > 0].index, 'total_population'].sum()
        })
    
    return pd.DataFrame(proposed_stops), pd.DataFrame(impact_analysis)

def create_proposed_stops_map(distances_df, gtfs_df, proposed_stops_df):
    """Create a map showing existing and proposed transit stops"""
    # Calculate map center
    center_lat = (gtfs_df['stop_lat'].mean() + distances_df['latitude'].mean()) / 2
    center_lon = (gtfs_df['stop_lon'].mean() + distances_df['longitude'].mean()) / 2
    
    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add existing transit stops
    for idx, row in gtfs_df.head(30).iterrows():  # Limit for performance
        folium.Marker(
            [row['stop_lat'], row['stop_lon']],
            popup=f"<b>Existing: {row['stop_name']}</b><br>ID: {row['stop_id']}",
            icon=folium.Icon(color='red', icon='train')
        ).add_to(m)
    
    # Add proposed stops
    for idx, row in proposed_stops_df.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"<b>Proposed: {row['proposed_stop_name']}</b><br>Tract: {row['tract_name']}<br>Equity Score: {row['current_equity_score']:.3f}<br>Population: {row['population_served']:,}",
            icon=folium.Icon(color='green', icon='plus')
        ).add_to(m)
    
    return m

def compute_underserved_candidates(distances_df, weights, bottom_quantile: float = 0.3):
    """Compute equity scores and return bottom-quantile tracts as candidate points for new stops"""
    distances_with_equity = distances_df.copy()
    distances_with_equity['transit_equity_score'] = calculate_transit_equity_score(distances_with_equity, weights)
    threshold = distances_with_equity['transit_equity_score'].quantile(bottom_quantile)
    candidates_df = distances_with_equity[distances_with_equity['transit_equity_score'] <= threshold].copy()
    return distances_with_equity, candidates_df

def compute_tract_weights(distances_df: pd.DataFrame, strategy: str, income_multiplier: float = 1.8) -> np.ndarray:
    """Compute per-tract weights for optimization based on strategy.

    - Equity-first: emphasize low-income and high-need tracts.
    - Density-first: emphasize population density strongly.
    - Balanced: blend equity- and density-driven weights.
    """
    population = distances_df['total_population'].to_numpy().astype(float)
    # Base density (ensure availability)
    if 'population_density' not in distances_df.columns:
        density = (distances_df['total_population'] / distances_df.get('area_sq_km', 1.0)).to_numpy().astype(float)
    else:
        density = distances_df['population_density'].to_numpy().astype(float)

    income = distances_df['median_income'].to_numpy().astype(float)
    # Normalize to [0,1] to use as multipliers
    def safe_norm(arr):
        arr = arr.astype(float)
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        if max_v - min_v < 1e-9:
            return np.ones_like(arr)
        return (arr - min_v) / (max_v - min_v)

    density_norm = safe_norm(density)
    # Lower income => higher weight; use inverse normalized income
    income_inv = 1.0 / (income + 1000.0)
    income_norm = safe_norm(income_inv)

    if strategy == 'Equity-first':
        # boost low-income areas
        factor = 1.0 + (income_multiplier - 1.0) * income_norm
        weights_arr = population * factor
    elif strategy == 'Density-first':
        # boost dense areas
        factor = 0.5 + 1.5 * density_norm
        weights_arr = population * factor
    else:  # Balanced
        factor_income = 1.0 + (income_multiplier - 1.0) * income_norm
        factor_density = 0.5 + 1.5 * density_norm
        factor = 0.5 * factor_income + 0.5 * factor_density
        weights_arr = population * factor

    # Ensure non-negative and not all zeros
    weights_arr = np.where(weights_arr < 0, 0.0, weights_arr)
    if float(weights_arr.sum()) <= 0:
        weights_arr = population.copy()
    return weights_arr

def greedy_optimize_new_stops(distances_df, candidates_df, k: int, weights, strategy: str = 'Balanced', income_multiplier: float = 1.8):
    """Greedy facility-location: iteratively select k candidate centroids to minimize population-weighted avg distance.

    Returns selected proposals list and the final per-tract new distances array.
    """
    current_min_distance = distances_df['distance_to_nearest_stop_km'].to_numpy().astype(float)
    tract_latitudes = distances_df['latitude'].to_numpy()
    tract_longitudes = distances_df['longitude'].to_numpy()
    tract_weights = compute_tract_weights(distances_df, strategy=strategy, income_multiplier=income_multiplier)

    selected = []
    used_candidate_ids = set()

    for step in range(max(0, k)):
        best_idx = None
        best_weighted_avg = float('inf')
        best_new_min = None
        best_stats = None

        for idx, cand in candidates_df.iterrows():
            if idx in used_candidate_ids:
                continue

            cand_lat = float(cand['latitude'])
            cand_lon = float(cand['longitude'])

            # Compute distance from all tracts to this candidate
            cand_distances = np.array([
                haversine_distance(float(tract_latitudes[i]), float(tract_longitudes[i]), cand_lat, cand_lon)
                for i in range(len(tract_latitudes))
            ])

            new_min = np.minimum(current_min_distance, cand_distances)
            # Population-weighted average distance
            weighted_avg = float(np.average(new_min, weights=tract_weights)) if tract_weights.sum() > 0 else float(new_min.mean())

            # Track impact stats for this candidate
            improvement = current_min_distance - new_min
            impacted_mask = improvement > 1e-9
            impacted_pop = float(tract_weights[impacted_mask].sum())
            avg_distance_reduction = float(improvement[impacted_mask].mean()) if impacted_mask.any() else 0.0

            # Estimate equity improvement (average across tracts) if this stop is added alone at this step
            tmp_df = distances_df.copy()
            tmp_df['distance_to_nearest_stop_km'] = new_min
            new_equity = calculate_transit_equity_score(tmp_df, weights)
            curr_equity = calculate_transit_equity_score(distances_df, weights)
            equity_improvement_pct = float(((new_equity.mean() - curr_equity.mean()) / curr_equity.mean()) * 100) if curr_equity.mean() > 0 else 0.0

            if weighted_avg < best_weighted_avg:
                best_weighted_avg = weighted_avg
                best_idx = idx
                best_new_min = new_min
                best_stats = {
                    'proposed_stop_id': f"OPT_{len(selected)+1:03d}",
                    'proposed_stop_name': f"Optimized Stop {len(selected)+1}",
                    'latitude': cand_lat,
                    'longitude': cand_lon,
                    'tract_name': cand.get('tract_name', cand.get('tract_id', str(best_idx))),
                    'population_served': int(round(impacted_pop)),
                    'avg_distance_reduction_km': avg_distance_reduction,
                    'equity_improvement_pct': equity_improvement_pct
                }

        if best_idx is None:
            break

        # Select the best candidate for this iteration
        used_candidate_ids.add(best_idx)
        current_min_distance = best_new_min
        selected.append(best_stats)

    return selected, current_min_distance

def compute_before_after_metrics(distances_df, before_distances, after_distances, weights):
    """Compute key comparison metrics for before vs after scenario."""
    before_df = distances_df.copy()
    after_df = distances_df.copy()
    before_df['distance_to_nearest_stop_km'] = before_distances
    after_df['distance_to_nearest_stop_km'] = after_distances

    before_df['transit_equity_score'] = calculate_transit_equity_score(before_df, weights)
    after_df['transit_equity_score'] = calculate_transit_equity_score(after_df, weights)

    # Metrics
    def pct_pop_within_threshold(df, threshold_km=0.5):
        within = df['distance_to_nearest_stop_km'] <= threshold_km
        pop_within = df.loc[within, 'total_population'].sum()
        total_pop = df['total_population'].sum()
        return (pop_within / total_pop) * 100 if total_pop > 0 else 0.0

    metrics = {
        'avg_equity_score_before': float(before_df['transit_equity_score'].mean()),
        'avg_equity_score_after': float(after_df['transit_equity_score'].mean()),
        'pct_pop_within_0_5km_before': float(pct_pop_within_threshold(before_df, 0.5)),
        'pct_pop_within_0_5km_after': float(pct_pop_within_threshold(after_df, 0.5)),
        'avg_distance_before_km': float(before_df['distance_to_nearest_stop_km'].mean()),
        'avg_distance_after_km': float(after_df['distance_to_nearest_stop_km'].mean()),
        'transit_deserts_before': int((before_df['transit_equity_score'] < 0.3).sum()),
        'transit_deserts_after': int((after_df['transit_equity_score'] < 0.3).sum()),
    }

    return metrics, before_df, after_df

def create_before_after_maps(before_df, after_df, gtfs_df, proposed_stops):
    """Create two folium maps (before, after) with equity choropleth-style markers and stops overlays."""
    center_lat = (gtfs_df['stop_lat'].mean() + before_df['latitude'].mean()) / 2
    center_lon = (gtfs_df['stop_lon'].mean() + before_df['longitude'].mean()) / 2

    def color_from_equity(score: float) -> str:
        if score < 0.3:
            return 'red'
        if score < 0.6:
            return 'orange'
        if score < 0.8:
            return 'yellow'
        return 'green'

    # Before map
    m_before = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    # Existing stops (blue)
    for _, row in gtfs_df.iterrows():
        folium.Marker(
            [row['stop_lat'], row['stop_lon']],
            popup=f"<b>{row.get('stop_name', row['stop_id'])}</b>",
            icon=folium.Icon(color='blue', icon='train')
        ).add_to(m_before)
    # Equity markers
    for _, row in before_df.iterrows():
        color = color_from_equity(float(row['transit_equity_score']))
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=12,
            popup=f"<b>{row['tract_name']}</b><br>Equity: {row['transit_equity_score']:.3f}<br>Distance: {row['distance_to_nearest_stop_km']:.2f} km",
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m_before)

    # After map
    m_after = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    # Existing stops (blue)
    for _, row in gtfs_df.iterrows():
        folium.Marker(
            [row['stop_lat'], row['stop_lon']],
            popup=f"<b>{row.get('stop_name', row['stop_id'])}</b>",
            icon=folium.Icon(color='blue', icon='train')
        ).add_to(m_after)
    # Proposed stops (red star)
    for prop in proposed_stops:
        folium.Marker(
            [prop['latitude'], prop['longitude']],
            popup=f"<b>{prop['proposed_stop_name']}</b>",
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m_after)
    # Equity markers after
    for _, row in after_df.iterrows():
        color = color_from_equity(float(row['transit_equity_score']))
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=12,
            popup=f"<b>{row['tract_name']}</b><br>Equity: {row['transit_equity_score']:.3f}<br>Distance: {row['distance_to_nearest_stop_km']:.2f} km",
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m_after)

    return m_before, m_after

def simulate_routes_shortest_path(tracts_df: pd.DataFrame, gtfs_df: pd.DataFrame, proposed_stops: list, num_cases: int = 6) -> pd.DataFrame:
    """Simplified travel-time simulation: time = walk_time_to_nearest_stop + fixed in-vehicle time.

    - Walk time uses nearest stop distance (km) at 5 km/h walking speed.
    - In-vehicle time to hub mocked as 15 minutes.
    Returns a DataFrame of sampled tracts with before/after times and % improvement.
    """
    rng = np.random.default_rng(42)
    sample_size = min(max(1, num_cases), len(tracts_df))
    sample_indices = rng.choice(len(tracts_df), size=sample_size, replace=False)

    proposed_coords = [(s['latitude'], s['longitude']) for s in proposed_stops] if proposed_stops else []

    records = []
    for idx in sample_indices:
        tract_row = tracts_df.iloc[int(idx)]
        d_before_km = float(tract_row['distance_to_nearest_stop_km'])
        walk_speed_kmh = 5.0
        in_vehicle_to_hub_min = 15.0
        before_time = (d_before_km / walk_speed_kmh) * 60.0 + in_vehicle_to_hub_min

        # After: nearest of existing distance and any proposed
        if proposed_coords:
            d_to_props = [
                haversine_distance(float(tract_row['latitude']), float(tract_row['longitude']), float(plat), float(plon))
                for plat, plon in proposed_coords
            ]
            d_after_km = float(min(d_before_km, min(d_to_props)))
        else:
            d_after_km = d_before_km
        after_time = (d_after_km / walk_speed_kmh) * 60.0 + in_vehicle_to_hub_min

        improvement_pct = ((before_time - after_time) / before_time * 100.0) if before_time > 0 else 0.0

        records.append({
            'tract_name': tract_row['tract_name'],
            'before_time_min': before_time,
            'after_time_min': after_time,
            'improvement_pct': improvement_pct
        })

    return pd.DataFrame(records).sort_values('improvement_pct', ascending=False)

def draw_routes_on_map(m: folium.Map, routes_df: pd.DataFrame, tracts_df: pd.DataFrame, gtfs_df: pd.DataFrame, proposed_stops: list | None, color: str):
    """Draw straight-line paths from tracts to nearest stop (existing only if proposed_stops is None, else to nearest of existing+proposed)."""
    # Prepare coordinates
    gtfs_coords = [(float(r['stop_lat']), float(r['stop_lon'])) for _, r in gtfs_df.iterrows()]
    proposed_coords = [(float(s['latitude']), float(s['longitude'])) for s in (proposed_stops or [])]

    def nearest_coord(lat: float, lon: float) -> tuple[float, float]:
        choices = gtfs_coords + proposed_coords if proposed_stops else gtfs_coords
        best = min(choices, key=lambda c: haversine_distance(lat, lon, c[0], c[1])) if choices else (lat, lon)
        return best

    for _, case in routes_df.iterrows():
        tract = tracts_df[tracts_df['tract_name'] == case['tract_name']].iloc[0]
        lat, lon = float(tract['latitude']), float(tract['longitude'])
        end_lat, end_lon = nearest_coord(lat, lon)
        folium.PolyLine([[lat, lon], [end_lat, end_lon]], color=color, weight=2, opacity=0.6).add_to(m)
    return m

def generate_pdf_report(distances_df, gtfs_df, cluster_stats=None, proposed_stops_df=None, impact_analysis_df=None):
    """Generate a comprehensive PDF report of the transit equity analysis"""
    
    # Create a buffer to store the PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1f77b4')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#2c3e50')
    )
    normal_style = styles['Normal']
    
    # Title page
    story.append(Paragraph("TransitFair: Transit Equity Analysis Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 30))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        f"This report analyzes transit access equity across {len(distances_df)} census tracts. "
        f"The analysis identifies areas with limited transit access and proposes solutions to improve "
        f"transportation equity for underserved communities.",
        normal_style
    ))
    story.append(Spacer(1, 20))
    
    # Key Metrics
    story.append(Paragraph("Key Metrics", heading_style))
    
    # Calculate key metrics
    avg_distance = distances_df['distance_to_nearest_stop_km'].mean()
    max_distance = distances_df['distance_to_nearest_stop_km'].max()
    transit_deserts = len(distances_df[distances_df['distance_to_nearest_stop_km'] > 1.0])
    avg_income = distances_df['median_income'].mean()
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Census Tracts Analyzed', f"{len(distances_df)}"],
        ['Average Distance to Transit', f"{avg_distance:.2f} km"],
        ['Maximum Distance to Transit', f"{max_distance:.2f} km"],
        ['Transit Deserts (>1km)', f"{transit_deserts}"],
        ['Average Median Income', f"${avg_income:,.0f}"],
        ['Total Population', f"{distances_df['total_population'].sum():,}"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Distance Analysis
    story.append(Paragraph("Distance Analysis", heading_style))
    story.append(Paragraph(
        f"The analysis reveals significant disparities in transit access. "
        f"While some areas have excellent transit coverage (minimum distance: {distances_df['distance_to_nearest_stop_km'].min():.2f} km), "
        f"others face substantial barriers with distances up to {max_distance:.2f} km from the nearest transit stop. "
        f"This creates {transit_deserts} transit deserts where residents have limited access to public transportation.",
        normal_style
    ))
    story.append(Spacer(1, 20))
    
    # Income Analysis
    story.append(Paragraph("Income and Transit Access", heading_style))
    
    # Calculate income quartiles and their average distances
    income_quartiles = pd.qcut(distances_df['median_income'], 4, labels=['Lowest', 'Second', 'Third', 'Highest'])
    income_distance_analysis = distances_df.groupby(income_quartiles)['distance_to_nearest_stop_km'].agg(['mean', 'count']).round(3)
    
    income_data = [['Income Quartile', 'Average Distance (km)', 'Number of Tracts']]
    for quartile, row in income_distance_analysis.iterrows():
        income_data.append([quartile, f"{row['mean']:.2f}", f"{row['count']}"])
    
    income_table = Table(income_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    income_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(income_table)
    story.append(Spacer(1, 20))
    
    # Cluster Analysis (if available)
    if cluster_stats is not None:
        story.append(Paragraph("Neighborhood Clustering Analysis", heading_style))
        story.append(Paragraph(
            "The analysis identified distinct neighborhood types based on transit access, income levels, "
            "and demographic characteristics. This clustering helps identify patterns and prioritize interventions.",
            normal_style
        ))
        story.append(Spacer(1, 20))
        
        # Convert cluster stats to table format
        cluster_data = [['Cluster Type', 'Count', 'Avg Distance (km)', 'Avg Income ($)', 'Avg Population']]
        for cluster_type, stats in cluster_stats.iterrows():
            cluster_data.append([
                cluster_type,
                f"{stats['count']}",
                f"{stats['distance_to_nearest_stop_km']:.2f}",
                f"${stats['median_income']:,.0f}",
                f"{stats['total_population']:,.0f}"
            ])
        
        cluster_table = Table(cluster_data, colWidths=[1.2*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        cluster_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(cluster_table)
        story.append(Spacer(1, 20))
    
    # New Stop Proposals (if available)
    if proposed_stops_df is not None and len(proposed_stops_df) > 0:
        story.append(Paragraph("Proposed New Transit Stops", heading_style))
        story.append(Paragraph(
            f"Based on the equity analysis, {len(proposed_stops_df)} new transit stops are proposed "
            "to improve access in underserved areas. These locations were selected based on low equity scores, "
            "high population density, and potential for maximum impact.",
            normal_style
        ))
        story.append(Spacer(1, 20))
        
        # Proposed stops table
        proposals_data = [['Proposed Stop', 'Tract', 'Current Equity Score', 'Population Served', 'Current Distance (km)']]
        for idx, row in proposed_stops_df.iterrows():
            proposals_data.append([
                row['proposed_stop_name'][:30] + "..." if len(row['proposed_stop_name']) > 30 else row['proposed_stop_name'],
                row['tract_name'][:20] + "..." if len(row['tract_name']) > 20 else row['tract_name'],
                f"{row['current_equity_score']:.3f}",
                f"{row['population_served']:,}",
                f"{row['current_distance']:.2f}"
            ])
        
        proposals_table = Table(proposals_data, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
        proposals_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(proposals_table)
        story.append(Spacer(1, 20))
        
        # Impact analysis
        if impact_analysis_df is not None:
            story.append(Paragraph("Expected Impact of New Stops", heading_style))
            
            total_pop_impacted = impact_analysis_df['total_population_impacted'].sum()
            avg_improvement = impact_analysis_df['equity_improvement_pct'].mean()
            
            story.append(Paragraph(
                f"The proposed new stops would impact approximately {total_pop_impacted:,} residents "
                f"with an average equity score improvement of {avg_improvement:.1f}%. "
                "This represents a significant step toward reducing transit access disparities.",
                normal_style
            ))
            story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Policy Recommendations", heading_style))
    story.append(Paragraph(
        "Based on this analysis, the following recommendations are proposed to improve transit equity:",
        normal_style
    ))
    story.append(Spacer(1, 12))
    
    recommendations = [
        "1. Prioritize new transit infrastructure in areas with equity scores below 0.3",
        "2. Implement targeted subsidies for low-income residents in transit deserts",
        "3. Develop partnerships with ride-sharing services for last-mile connectivity",
        "4. Establish community transportation programs in underserved areas",
        "5. Conduct regular equity assessments to monitor progress",
        "6. Engage with community stakeholders to identify local transportation needs"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
        story.append(Spacer(1, 6))
    
    story.append(Spacer(1, 20))
    
    # Methodology
    story.append(Paragraph("Methodology", heading_style))
    story.append(Paragraph(
        "This analysis used the following methodology:",
        normal_style
    ))
    story.append(Spacer(1, 12))
    
    methodology = [
        "• Distance calculations using the Haversine formula for geographic accuracy",
        "• K-means clustering to identify neighborhood types based on multiple factors",
        "• Transit equity scoring combining population density, distance, and income",
        "• Machine learning algorithms to identify optimal locations for new stops",
        "• Statistical analysis to measure disparities and track improvements"
    ]
    
    for method in methodology:
        story.append(Paragraph(method, normal_style))
        story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_visualization_for_pdf(distances_df, plot_type='distance_histogram'):
    """Create matplotlib visualizations for PDF inclusion"""
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'distance_histogram':
        plt.hist(distances_df['distance_to_nearest_stop_km'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Distances to Transit Stops', fontsize=14, fontweight='bold')
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('Number of Census Tracts', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    elif plot_type == 'income_distance_scatter':
        plt.scatter(distances_df['distance_to_nearest_stop_km'], distances_df['median_income'], 
                   alpha=0.6, c='red', s=50)
        plt.title('Distance to Transit vs Median Income', fontsize=14, fontweight='bold')
        plt.xlabel('Distance to Nearest Transit Stop (km)', fontsize=12)
        plt.ylabel('Median Income ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    elif plot_type == 'equity_histogram' and 'transit_equity_score' in distances_df.columns:
        plt.hist(distances_df['transit_equity_score'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.title('Distribution of Transit Equity Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Equity Score', fontsize=12)
        plt.ylabel('Number of Census Tracts', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚌 TransitFair: Mapping Transit Inequity</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze transit access disparities across neighborhoods using real NYC data")
    
    # Sidebar
    st.sidebar.title("📊 Data Upload")
    st.sidebar.markdown("---")
    
    # File uploaders
    st.sidebar.subheader("📁 Upload Your Data")
    
    # Option to use sample data or upload custom data
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use Real NYC Data (Pre-loaded)", "Upload Custom Data"]
    )
    
    if data_option == "Use Real NYC Data (Pre-loaded)":
        st.sidebar.success("✅ Using real NYC MTA and Census data")
        gtfs_df, census_df = load_real_data()
        data_loaded = True
    else:
        gtfs_df, census_df = None, None
        data_loaded = False
        
        # File uploaders for custom data
        st.sidebar.markdown("#### Transit Data (GTFS)")
        gtfs_file = st.sidebar.file_uploader(
            "Upload transit stops CSV",
            type=['csv'],
            help="CSV with columns: stop_id, stop_name, stop_lat, stop_lon"
        )
        
        st.sidebar.markdown("#### Census Data")
        census_file = st.sidebar.file_uploader(
            "Upload census data CSV",
            type=['csv'],
            help="CSV with columns: tract_id, latitude, longitude, total_population, median_income, car_ownership_rate"
        )
        
        if gtfs_file and census_file:
            try:
                gtfs_df = pd.read_csv(gtfs_file)
                census_df = pd.read_csv(census_file)
                data_loaded = True
                st.sidebar.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")
                data_loaded = False
    
    # Main content
    if data_loaded and gtfs_df is not None and census_df is not None:
        display_data_preview(gtfs_df, census_df)
        
        # Check required columns
        if validate_data(gtfs_df, census_df):
            st.success("✅ Data validation passed! Ready for analysis.")
            
            # Calculate distances (STEP 2)
            with st.spinner("🔄 Calculating distances to transit stops..."):
                distances_df = calculate_distances_to_transit(gtfs_df, census_df)
            
            st.success(f"✅ Distance calculations complete! Analyzed {len(distances_df)} census tracts.")
            
            # STEP 7: Enhanced Dashboard with View Toggle and Sidebar Controls
            st.sidebar.markdown("---")
            st.sidebar.title("🎛️ Analysis Controls")
            
            # View selector
            analysis_view = st.sidebar.selectbox(
                "Choose Analysis View:",
                ["🗺️ Geospatial Analysis", "📊 Data Exploration", "🔍 Transit Equity Analysis", "📈 Advanced Analytics", "🧪 Before/After Optimization"],
                help="Select which analysis view to display"
            )
            
            # Equity score weight sliders
            st.sidebar.markdown("### ⚖️ Equity Score Weights")
            st.sidebar.markdown("Adjust the importance of each factor in the transit equity score:")
            
            pop_weight = st.sidebar.slider(
                "Population Density Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.1,
                help="Higher values prioritize areas with more people"
            )
            
            dist_weight = st.sidebar.slider(
                "Distance Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.1,
                help="Higher values prioritize areas closer to transit"
            )
            
            income_weight = st.sidebar.slider(
                "Income Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Higher values prioritize lower-income areas"
            )
            
            # Normalize weights to sum to 1
            total_weight = pop_weight + dist_weight + income_weight
            if total_weight > 0:
                pop_weight = pop_weight / total_weight
                dist_weight = dist_weight / total_weight
                income_weight = income_weight / total_weight
            else:
                pop_weight = dist_weight = income_weight = 1/3
            
            # Display current weights
            st.sidebar.markdown(f"**Current Weights:**")
            st.sidebar.markdown(f"Population: {pop_weight:.1%}")
            st.sidebar.markdown(f"Distance: {dist_weight:.1%}")
            st.sidebar.markdown(f"Income: {income_weight:.1%}")
            
            # Clustering controls
            st.sidebar.markdown("### 🎯 Clustering Settings")
            n_clusters = st.sidebar.slider(
                "Number of Clusters",
                min_value=2,
                max_value=6,
                value=4,
                help="Number of groups to divide neighborhoods into"
            )
            
            # Equity threshold for new stops
            st.sidebar.markdown("### 🚏 New Stop Proposals")
            equity_threshold = st.sidebar.slider(
                "Equity Threshold",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Areas below this equity score will be considered for new stops"
            )
            
            max_proposals = st.sidebar.slider(
                "Max Proposals",
                min_value=3,
                max_value=10,
                value=5,
                help="Maximum number of new stop proposals to generate"
            )
            
            # Download section
            st.sidebar.markdown("---")
            st.sidebar.title("💾 Download Data")
            
            # Create weights dictionary for equity score calculation
            weights = {
                'population_density': pop_weight,
                'distance': dist_weight,
                'income': income_weight
            }
            
            # Display selected view
            if analysis_view == "🗺️ Geospatial Analysis":
                display_geospatial_analysis(gtfs_df, census_df, distances_df)
            elif analysis_view == "📊 Data Exploration":
                display_data_exploration(gtfs_df, census_df, distances_df)
            elif analysis_view == "🔍 Transit Equity Analysis":
                display_transit_equity_analysis(gtfs_df, census_df, distances_df, weights)
            elif analysis_view == "📈 Advanced Analytics":
                display_advanced_analytics(gtfs_df, census_df, distances_df, n_clusters, equity_threshold, max_proposals, weights)
            elif analysis_view == "🧪 Before/After Optimization":
                display_before_after_optimization(gtfs_df, distances_df, weights, max_proposals)
            
            # Global download buttons
            st.markdown("---")
            st.markdown("### 💾 Download Processed Data")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Download processed distances data
                csv_distances = distances_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Distance Analysis",
                    data=csv_distances,
                    file_name="transit_distance_analysis.csv",
                    mime="text/csv",
                    help="Download the complete distance analysis with all calculated metrics"
                )
            
            with col2:
                # Download transit stops data
                csv_stops = gtfs_df.to_csv(index=False)
                st.download_button(
                    label="🚏 Download Transit Stops",
                    data=csv_stops,
                    file_name="transit_stops.csv",
                    mime="text/csv",
                    help="Download the transit stops data used in analysis"
                )
            
            with col3:
                # Download census data
                csv_census = census_df.to_csv(index=False)
                st.download_button(
                    label="🏘️ Download Census Data",
                    data=csv_census,
                    file_name="census_data.csv",
                    mime="text/csv",
                    help="Download the census demographic data used in analysis"
                )
            
            with col4:
                # PDF Report Generation
                st.markdown("**📄 Generate PDF Report**")
                if st.button("📋 Create Comprehensive Report", help="Generate a detailed PDF report with all analysis results"):
                    with st.spinner("🔄 Generating comprehensive PDF report..."):
                        try:
                            # Get cluster stats and proposals if available
                            cluster_stats = None
                            proposed_stops_df = None
                            impact_analysis_df = None
                            
                            # Try to get cluster stats from advanced analytics
                            if 'transit_equity_score' in distances_df.columns:
                                # Calculate equity scores if not already done
                                equity_scores = calculate_transit_equity_score(distances_df, weights)
                                distances_with_equity = distances_df.copy()
                                distances_with_equity['transit_equity_score'] = equity_scores
                                
                                # Try to get clustering data
                                try:
                                    normalized_df, scaler, feature_columns = normalize_features(distances_with_equity)
                                    cluster_labels, kmeans = perform_kmeans_clustering(normalized_df, n_clusters)
                                    distances_with_clusters, cluster_stats, cluster_to_equity = analyze_clusters(distances_with_equity, cluster_labels)
                                    
                                    # Try to get proposal data
                                    try:
                                        proposed_stops_df, impact_analysis_df = propose_new_stops(distances_with_equity, gtfs_df, equity_threshold, max_proposals)
                                    except:
                                        pass  # Proposals not available
                                except:
                                    pass  # Clustering not available
                            
                            # Generate PDF
                            pdf_buffer = generate_pdf_report(distances_df, gtfs_df, cluster_stats, proposed_stops_df, impact_analysis_df)
                            
                            st.success("✅ PDF report generated successfully!")
                            
                            # Download button for PDF
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"transit_equity_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                help="Download the comprehensive PDF report with all analysis results"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating PDF report: {str(e)}")
                            st.info("💡 Try running the Advanced Analytics view first to ensure all data is available.")
            
            # Explanatory sections
            st.markdown("---")
            with st.expander("📚 Understanding the Analysis", expanded=False):
                st.markdown("""
                ### 🎯 What are Clusters?
                **Clusters** are groups of neighborhoods with similar characteristics based on:
                - Distance to transit stops
                - Median household income
                - Car ownership rates
                - Population density
                
                **Cluster Types:**
                - 🚨 **Severely Underserved**: High distance to transit, low income
                - ⚠️ **Moderately Underserved**: Moderate distance, moderate income
                - ✅ **Adequately Served**: Reasonable access, moderate income
                - 🌟 **Well Served**: Good transit access, higher income
                
                ### 🏜️ What is a Transit Desert?
                A **transit desert** is an area where:
                - Residents have limited access to public transportation
                - Walking distance to transit stops is >0.5 miles (0.8 km)
                - Often correlates with lower-income neighborhoods
                - Creates transportation inequity
                
                ### 💯 Transit Equity Score
                The **equity score** combines multiple factors to identify areas most in need of transit improvements:
                - **Population Density**: Areas with more people need better access
                - **Distance to Transit**: Closer is better (inverse relationship)
                - **Income Level**: Lower-income areas often have fewer transportation options
                
                **Formula**: `Equity Score = (Population Weight × Normalized Density) + (Distance Weight × Inverse Distance) + (Income Weight × Inverse Income)`
                
                ### 🚏 New Stop Proposals
                The system identifies optimal locations for new transit stops by:
                1. Finding areas with low equity scores
                2. Calculating the impact of adding stops at tract centroids
                3. Estimating improvements in access and equity scores
                4. Prioritizing areas that would benefit most people
                """)
            
            with st.expander("🔧 How to Use This Tool", expanded=False):
                st.markdown("""
                ### 📊 Getting Started
                1. **Data Upload**: Use the pre-loaded NYC data or upload your own CSV files
                2. **View Selection**: Choose from different analysis perspectives in the sidebar
                3. **Parameter Adjustment**: Use sliders to customize equity weights and clustering
                4. **Download Results**: Export processed data for further analysis
                
                ### 🎛️ Customizing Analysis
                - **Equity Weights**: Adjust the importance of population, distance, and income
                - **Clusters**: Change the number of neighborhood groups (2-6)
                - **Proposals**: Set thresholds for new stop recommendations
                
                ### 📈 Interpreting Results
                - **Red areas** on maps typically indicate high need/low access
                - **Green areas** show good transit access
                - **Cluster colors** help identify patterns across neighborhoods
                - **Equity scores** range from 0 (lowest need) to 1 (highest need)
                
                ### 🎯 Policy Applications
                - **Transit Planning**: Identify priority areas for new routes/stops
                - **Equity Analysis**: Measure transportation access disparities
                - **Resource Allocation**: Target investments where they're needed most
                - **Community Engagement**: Visualize and communicate transit needs
                """)
        else:
            st.error("❌ Data validation failed. Please check your file formats.")
    else:
        display_welcome_screen()

def load_real_data():
    """Load the real NYC data we processed"""
    try:
        # Load the processed real data
        gtfs_df = pd.read_csv("data/real_gtfs_stops_processed.csv")
        census_df = pd.read_csv("data/real_census_tracts_processed.csv")
        return gtfs_df, census_df
    except FileNotFoundError:
        st.error("Real data files not found. Please ensure the data files are in the correct location.")
        return None, None

def validate_data(gtfs_df, census_df):
    """Validate that the uploaded data has required columns"""
    required_gtfs_cols = ['stop_id', 'stop_lat', 'stop_lon']
    required_census_cols = ['tract_id', 'latitude', 'longitude', 'total_population', 'median_income']
    
    gtfs_valid = all(col in gtfs_df.columns for col in required_gtfs_cols)
    census_valid = all(col in census_df.columns for col in required_census_cols)
    
    return gtfs_valid and census_valid

def display_welcome_screen():
    """Display welcome screen when no data is loaded"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Welcome to TransitFair!</h3>
        <p>This application analyzes transit access disparities across neighborhoods using real data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 What you can do:")
        
        st.markdown("""
        - **Upload your own data**: GTFS transit stops and Census demographic data
        - **Use real NYC data**: Pre-loaded NYC MTA subway stops and Census tracts
        - **Analyze transit equity**: Calculate distances, identify underserved areas
        - **Visualize results**: Interactive maps and charts
        - **Generate insights**: Transit equity scores and recommendations
        """)
        
        st.markdown("### 🚀 Getting Started:")
        st.markdown("""
        1. **Choose data source** in the sidebar
        2. **Upload your files** or use pre-loaded NYC data
        3. **Explore the analysis** across different tabs
        4. **Generate insights** about transit equity
        """)
        
        # Show sample data structure
        st.markdown("### 📊 Expected Data Format:")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Transit Data (GTFS)**")
            st.code("""
stop_id,stop_name,stop_lat,stop_lon
101,Central Station,40.7128,-74.0060
102,Downtown Hub,40.7589,-73.9851
            """)
        
        with col_b:
            st.markdown("**Census Data**")
            st.code("""
tract_id,latitude,longitude,total_population,median_income
36061000100,40.7128,-74.0060,8500,45000
36061000200,40.7505,-73.9934,12000,65000
            """)

def display_data_preview(gtfs_df, census_df):
    """Display preview of uploaded data"""
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📊 Data Preview</h2>', unsafe_allow_html=True)
    
    # Data summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚇 Transit Stops", len(gtfs_df))
    
    with col2:
        st.metric("🏘️ Census Tracts", len(census_df))
    
    with col3:
        if 'stop_lat' in gtfs_df.columns:
            lat_range = f"{gtfs_df['stop_lat'].min():.3f} - {gtfs_df['stop_lat'].max():.3f}"
            st.metric("📍 Latitude Range", lat_range)
    
    with col4:
        if 'total_population' in census_df.columns:
            total_pop = f"{census_df['total_population'].sum():,}"
            st.metric("👥 Total Population", total_pop)
    
    # Data previews
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚇 Transit Stops Preview")
        st.dataframe(gtfs_df.head(), use_container_width=True)
        
        if st.checkbox("Show transit data info"):
            st.write("**Columns:**", list(gtfs_df.columns))
            st.write("**Data types:**", gtfs_df.dtypes.to_dict())
    
    with col2:
        st.subheader("🏘️ Census Data Preview")
        st.dataframe(census_df.head(), use_container_width=True)
        
        if st.checkbox("Show census data info"):
            st.write("**Columns:**", list(census_df.columns))
            st.write("**Data types:**", census_df.dtypes.to_dict())

def display_geospatial_analysis(gtfs_df, census_df, distances_df):
    """Display geospatial analysis and maps"""
    st.markdown('<h2 class="sub-header">🗺️ Geospatial Analysis</h2>', unsafe_allow_html=True)
    
    # Distance analysis summary
    st.subheader("📏 Distance Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_distance = distances_df['distance_to_nearest_stop_km'].mean()
        st.metric("📏 Avg Distance to Transit", f"{avg_distance:.2f} km")
    
    with col2:
        max_distance = distances_df['distance_to_nearest_stop_km'].max()
        st.metric("🚶 Max Distance", f"{max_distance:.2f} km")
    
    with col3:
        min_distance = distances_df['distance_to_nearest_stop_km'].min()
        st.metric("🏃 Min Distance", f"{min_distance:.2f} km")
    
    with col4:
        transit_deserts = len(distances_df[distances_df['distance_to_nearest_stop_km'] > 1.0])
        st.metric("🏜️ Transit Deserts (>1km)", transit_deserts)
    
    # Create interactive map with distance information
    st.subheader("📍 Interactive Map: Transit Access Analysis")
    
    # Calculate map center
    if 'stop_lat' in gtfs_df.columns and 'latitude' in census_df.columns:
        center_lat = (gtfs_df['stop_lat'].mean() + census_df['latitude'].mean()) / 2
        center_lon = (gtfs_df['stop_lon'].mean() + census_df['longitude'].mean()) / 2
        
        # Create Folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add transit stops
        for idx, row in gtfs_df.head(50).iterrows():  # Limit to first 50 for performance
            folium.Marker(
                [row['stop_lat'], row['stop_lon']],
                popup=f"<b>{row['stop_name']}</b><br>ID: {row['stop_id']}",
                icon=folium.Icon(color='red', icon='train')
            ).add_to(m)
        
        # Add census tract centroids with distance coloring
        for idx, row in distances_df.head(30).iterrows():  # Limit to first 30 for performance
            distance = row['distance_to_nearest_stop_km']
            
            # Color based on distance (green=close, yellow=medium, red=far)
            if distance <= 0.5:
                color = 'green'
            elif distance <= 1.0:
                color = 'orange'
            else:
                color = 'red'
            
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=10,
                popup=f"<b>{row['tract_name']}</b><br>Population: {row['total_population']:,}<br>Income: ${row['median_income']:,}<br>Distance: {distance:.2f} km<br>Nearest: {row['nearest_stop_name']}",
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        st_folium(m, width=800, height=600)
    
    # Distance distribution
    st.subheader("📊 Distance Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distance histogram
        fig_distance = px.histogram(
            distances_df,
            x='distance_to_nearest_stop_km',
            title='Distribution of Distances to Nearest Transit Stop',
            labels={'distance_to_nearest_stop_km': 'Distance (km)', 'count': 'Number of Tracts'},
            nbins=20
        )
        st.plotly_chart(fig_distance, use_container_width=True)
    
    with col2:
        # Distance vs Income scatter
        fig_scatter = px.scatter(
            distances_df,
            x='distance_to_nearest_stop_km',
            y='median_income',
            title='Distance to Transit vs Median Income',
            labels={'distance_to_nearest_stop_km': 'Distance (km)', 'median_income': 'Median Income ($)'},
            hover_data=['tract_name', 'total_population']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top transit deserts
    st.subheader("🏜️ Top Transit Deserts (Farthest from Transit)")
    
    transit_deserts_df = distances_df.nlargest(10, 'distance_to_nearest_stop_km')
    st.dataframe(
        transit_deserts_df[['tract_name', 'distance_to_nearest_stop_km', 'nearest_stop_name', 'total_population', 'median_income']].round(2),
        use_container_width=True
    )

def display_data_exploration(gtfs_df, census_df, distances_df):
    """Display data exploration and statistics"""
    st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    # Census data analysis
    st.subheader("🏘️ Census Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'total_population' in census_df.columns:
            # Population distribution
            fig_pop = px.histogram(
                census_df, 
                x='total_population',
                title='Population Distribution by Census Tract',
                labels={'total_population': 'Population', 'count': 'Number of Tracts'}
            )
            st.plotly_chart(fig_pop, use_container_width=True)
    
    with col2:
        if 'median_income' in census_df.columns:
            # Income distribution
            fig_income = px.histogram(
                census_df, 
                x='median_income',
                title='Income Distribution by Census Tract',
                labels={'median_income': 'Median Income ($)', 'count': 'Number of Tracts'}
            )
            st.plotly_chart(fig_income, use_container_width=True)
    
    # Distance analysis
    st.subheader("📏 Distance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distance vs Population
        fig_dist_pop = px.scatter(
            distances_df,
            x='distance_to_nearest_stop_km',
            y='total_population',
            title='Distance vs Population',
            labels={'distance_to_nearest_stop_km': 'Distance (km)', 'total_population': 'Population'},
            hover_data=['tract_name']
        )
        st.plotly_chart(fig_dist_pop, use_container_width=True)
    
    with col2:
        # Distance vs Car Ownership
        if 'car_ownership_rate' in distances_df.columns:
            fig_dist_car = px.scatter(
                distances_df,
                x='distance_to_nearest_stop_km',
                y='car_ownership_rate',
                title='Distance vs Car Ownership Rate',
                labels={'distance_to_nearest_stop_km': 'Distance (km)', 'car_ownership_rate': 'Car Ownership Rate'},
                hover_data=['tract_name']
            )
            st.plotly_chart(fig_dist_car, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    if all(col in distances_df.columns for col in ['total_population', 'median_income', 'car_ownership_rate', 'distance_to_nearest_stop_km']):
        # Select numeric columns for correlation
        numeric_cols = ['total_population', 'median_income', 'car_ownership_rate', 'distance_to_nearest_stop_km']
        if 'population_density' in distances_df.columns:
            numeric_cols.append('population_density')
        
        correlation_matrix = distances_df[numeric_cols].corr()
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title='Correlation Matrix of Variables',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Show correlation values
        st.write("**Correlation Values:**")
        st.dataframe(correlation_matrix.round(3))

def display_transit_equity_analysis(gtfs_df, census_df, distances_df, weights):
    """Display transit equity analysis"""
    st.markdown('<h2 class="sub-header">🔍 Transit Equity Analysis</h2>', unsafe_allow_html=True)
    
    st.info("🎯 This section analyzes transit access equity using distance calculations, demographic data, and equity scoring.")
    
    # Equity Score Calculation Section
    st.subheader("💯 STEP 5: Transit Equity Score")
    
    # Display current weights from sidebar
    st.markdown("**⚖️ Current Equity Score Weights (from sidebar):**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Population Density", f"{weights['population_density']:.1%}")
    
    with col2:
        st.metric("Distance to Transit", f"{weights['distance']:.1%}")
    
    with col3:
        st.metric("Income Level", f"{weights['income']:.1%}")
    
    # Calculate equity scores
    with st.spinner("🔄 Calculating transit equity scores..."):
        equity_scores = calculate_transit_equity_score(distances_df, weights)
        distances_with_equity = distances_df.copy()
        distances_with_equity['transit_equity_score'] = equity_scores
    
    st.success(f"✅ Equity scores calculated! Average score: {equity_scores.mean():.3f}")
    
    # Equity Score Analysis
    st.subheader("📊 Equity Score Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_equity = distances_with_equity['transit_equity_score'].mean()
        st.metric("📊 Average Equity Score", f"{avg_equity:.3f}")
    
    with col2:
        low_equity = len(distances_with_equity[distances_with_equity['transit_equity_score'] < 0.3])
        st.metric("🚨 Low Equity (<0.3)", low_equity)
    
    with col3:
        high_equity = len(distances_with_equity[distances_with_equity['transit_equity_score'] > 0.7])
        st.metric("🌟 High Equity (>0.7)", high_equity)
    
    with col4:
        equity_range = distances_with_equity['transit_equity_score'].max() - distances_with_equity['transit_equity_score'].min()
        st.metric("📏 Equity Range", f"{equity_range:.3f}")
    
    # Equity Score Distribution
    st.subheader("📈 Equity Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Equity score histogram
        fig_equity_hist = px.histogram(
            distances_with_equity,
            x='transit_equity_score',
            title='Distribution of Transit Equity Scores',
            labels={'transit_equity_score': 'Equity Score', 'count': 'Number of Tracts'},
            nbins=20
        )
        st.plotly_chart(fig_equity_hist, use_container_width=True)
    
    with col2:
        # Equity vs Distance scatter
        fig_equity_dist = px.scatter(
            distances_with_equity,
            x='distance_to_nearest_stop_km',
            y='transit_equity_score',
            title='Equity Score vs Distance to Transit',
            labels={'distance_to_nearest_stop_km': 'Distance (km)', 'transit_equity_score': 'Equity Score'},
            hover_data=['tract_name', 'total_population', 'median_income']
        )
        st.plotly_chart(fig_equity_dist, use_container_width=True)
    
    # Choropleth Map
    st.subheader("🗺️ Transit Equity Choropleth Map")
    
    with st.spinner("🔄 Creating equity choropleth map..."):
        equity_map = create_equity_choropleth_map(distances_with_equity, gtfs_df)
    
    st_folium(equity_map, width=800, height=600)
    
    # Equity Score Rankings
    st.subheader("🏆 Transit Equity Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚨 Lowest Equity Scores (Priority Areas):**")
        lowest_equity = distances_with_equity.nsmallest(10, 'transit_equity_score')
        st.dataframe(
            lowest_equity[['tract_name', 'transit_equity_score', 'distance_to_nearest_stop_km', 'median_income', 'total_population']].round(3),
            use_container_width=True
        )
    
    with col2:
        st.markdown("**🌟 Highest Equity Scores (Well-Served Areas):**")
        highest_equity = distances_with_equity.nlargest(10, 'transit_equity_score')
        st.dataframe(
            highest_equity[['tract_name', 'transit_equity_score', 'distance_to_nearest_stop_km', 'median_income', 'total_population']].round(3),
            use_container_width=True
        )
    
    # Equity Score Components Analysis
    st.subheader("🔍 Equity Score Components Analysis")
    
    # Show how each component contributes to the final score
    st.markdown("**📊 Equity Score Formula:**")
    st.latex(f"Equity Score = {weights['population_density']:.1f} \\times Population Density + {weights['distance']:.1f} \\times (1/Distance) + {weights['income']:.1f} \\times (1/Income)")
    
    # Component correlation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Equity score vs population density
        fig_equity_pop = px.scatter(
            distances_with_equity,
            x='population_density',
            y='transit_equity_score',
            title='Equity Score vs Population Density',
            labels={'population_density': 'Population Density', 'transit_equity_score': 'Equity Score'},
            hover_data=['tract_name']
        )
        st.plotly_chart(fig_equity_pop, use_container_width=True)
    
    with col2:
        # Equity score vs income
        fig_equity_income = px.scatter(
            distances_with_equity,
            x='median_income',
            y='transit_equity_score',
            title='Equity Score vs Median Income',
            labels={'median_income': 'Median Income ($)', 'transit_equity_score': 'Equity Score'},
            hover_data=['tract_name']
        )
        st.plotly_chart(fig_equity_income, use_container_width=True)
    
    # Policy Recommendations
    st.subheader("📋 Policy Recommendations")
    
    # Calculate recommendations based on equity scores
    low_equity_tracts = distances_with_equity[distances_with_equity['transit_equity_score'] < 0.3]
    medium_equity_tracts = distances_with_equity[(distances_with_equity['transit_equity_score'] >= 0.3) & (distances_with_equity['transit_equity_score'] < 0.6)]
    
    st.markdown(f"""
    ### 🎯 Equity-Based Recommendations:
    
    **High Priority Areas (Equity Score < 0.3):**
    - **{len(low_equity_tracts)} tracts** need immediate attention
    - **Average distance to transit:** {low_equity_tracts['distance_to_nearest_stop_km'].mean():.2f} km
    - **Average income:** ${low_equity_tracts['median_income'].mean():,.0f}
    - **Recommendations:** New transit stops, improved service frequency, subsidized transit passes
    
    **Medium Priority Areas (Equity Score 0.3-0.6):**
    - **{len(medium_equity_tracts)} tracts** need moderate improvements
    - **Average distance to transit:** {medium_equity_tracts['distance_to_nearest_stop_km'].mean():.2f} km
    - **Recommendations:** Service optimization, better connections, accessibility improvements
    
    **Equity Score Insights:**
    - **Correlation with distance:** {distances_with_equity['transit_equity_score'].corr(distances_with_equity['distance_to_nearest_stop_km']):.3f}
    - **Correlation with income:** {distances_with_equity['transit_equity_score'].corr(distances_with_equity['median_income']):.3f}
    - **Correlation with population density:** {distances_with_equity['transit_equity_score'].corr(distances_with_equity['population_density']):.3f}
    """)
    
    # Download equity data
    st.subheader("📥 Download Equity Analysis Data")
    
    equity_csv = distances_with_equity.to_csv(index=False)
    st.download_button(
        label="📊 Download Equity Analysis Results (CSV)",
        data=equity_csv,
        file_name="transit_equity_analysis.csv",
        mime="text/csv"
    )

def display_before_after_optimization(gtfs_df, distances_df, weights, k_new_stops):
    """Side-by-side before/after maps and key comparisons using greedy optimized stops."""
    st.markdown('<h2 class="sub-header">🧪 Before/After Optimization</h2>', unsafe_allow_html=True)

    # Compute underserved candidates
    with st.spinner("🔄 Identifying underserved areas (bottom 30% equity score)..."):
        distances_with_equity, candidates_df = compute_underserved_candidates(distances_df, weights, bottom_quantile=0.3)

    if len(candidates_df) == 0:
        st.warning("No candidates found in the bottom 30% by equity score. Adjust weights or load different data.")
        return

    st.info(f"Selected {len(candidates_df)} candidate tracts from the lowest equity segment.")

    # Strategy selector
    strategy = st.radio(
        "Weighting Strategy",
        options=["Equity-first", "Density-first", "Balanced"],
        index=2,
        help="Choose how optimization weights tracts: by equity need, density, or both"
    )

    income_multiplier = st.slider(
        "Low-income emphasis (×)", min_value=1.0, max_value=2.5, value=1.8, step=0.1,
        help="Multiplier applied to low-income tracts under equity-first/balanced strategies"
    )

    # Optimize new stops (greedy)
    with st.spinner(f"🔄 Selecting up to {k_new_stops} optimized stop locations using {strategy} strategy..."):
        selected, after_min_dist = greedy_optimize_new_stops(
            distances_with_equity, candidates_df, k_new_stops, weights, strategy=strategy, income_multiplier=income_multiplier
        )

    if len(selected) == 0:
        st.warning("No optimized stops could be selected.")
        return

    # Compute metrics and before/after DFs
    before_dist = distances_with_equity['distance_to_nearest_stop_km'].to_numpy()
    metrics, before_df, after_df = compute_before_after_metrics(distances_with_equity, before_dist, after_min_dist, weights)

    # Maps side-by-side
    st.subheader("🗺️ Side-by-Side Maps")
    col1, col2 = st.columns(2)
    with st.spinner("🧭 Building maps..."):
        m_before, m_after = create_before_after_maps(before_df, after_df, gtfs_df, selected)
        # Build routes once for both views
        routes_df = simulate_routes_shortest_path(before_df, gtfs_df, selected, num_cases=6)
        # Draw on both maps before rendering
        m_before = draw_routes_on_map(m_before, routes_df, before_df, gtfs_df, proposed_stops=None, color='blue')
        m_after = draw_routes_on_map(m_after, routes_df, before_df, gtfs_df, proposed_stops=selected, color='purple')
    with col1:
        st.markdown("**Before Optimization**")
        st_folium(m_before, width=650, height=550)
    with col2:
        st.markdown("**After Optimization**")
        st_folium(m_after, width=650, height=550)

    # Key comparisons
    st.subheader("📊 Key Comparisons")
    comp_table = pd.DataFrame([
        {
            'Metric': 'Average Equity Score',
            'Before': f"{metrics['avg_equity_score_before']:.3f}",
            'After': f"{metrics['avg_equity_score_after']:.3f}"
        },
        {
            'Metric': '% Population within 0.5 km',
            'Before': f"{metrics['pct_pop_within_0_5km_before']:.1f}%",
            'After': f"{metrics['pct_pop_within_0_5km_after']:.1f}%"
        },
        {
            'Metric': 'Average Distance to Nearest Stop (km)',
            'Before': f"{metrics['avg_distance_before_km']:.2f}",
            'After': f"{metrics['avg_distance_after_km']:.2f}"
        },
        {
            'Metric': 'Number of Transit Deserts (Equity < 0.3)',
            'Before': f"{metrics['transit_deserts_before']}",
            'After': f"{metrics['transit_deserts_after']}"
        }
    ])
    st.dataframe(comp_table, use_container_width=True)

    # Impact table for proposed stops
    st.subheader("🚏 Proposed Optimized Stops (Impact)")
    impact_df = pd.DataFrame(selected)
    impact_df = impact_df.sort_values(by=['population_served', 'avg_distance_reduction_km', 'equity_improvement_pct'], ascending=False)
    st.dataframe(
        impact_df[['proposed_stop_name', 'latitude', 'longitude', 'population_served', 'avg_distance_reduction_km', 'equity_improvement_pct']].round(3),
        use_container_width=True
    )

    # Travel time comparison table
    st.subheader("⏱️ Case Study Travel Time Improvements")
    st.dataframe(
        routes_df[['tract_name', 'before_time_min', 'after_time_min', 'improvement_pct']].round(2),
        use_container_width=True
    )

def display_advanced_analytics(gtfs_df, census_df, distances_df, n_clusters, equity_threshold, max_proposals, weights):
    """Display advanced analytics and machine learning results"""
    st.markdown('<h2 class="sub-header">📈 Advanced Analytics & Machine Learning</h2>', unsafe_allow_html=True)
    
    st.info("🤖 This section performs feature engineering, normalization, K-means clustering analysis, and new stop proposals.")
    
    # Feature Engineering Section
    st.subheader("🔧 STEP 3: Feature Engineering")
    
    with st.spinner("🔄 Normalizing features for machine learning..."):
        normalized_df, scaler, feature_columns = normalize_features(distances_df)
    
    st.success(f"✅ Feature normalization complete! Normalized {len(feature_columns)} features.")
    
    # Show normalized features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Original Features:**")
        st.dataframe(
            distances_df[feature_columns].describe().round(3),
            use_container_width=True
        )
    
    with col2:
        st.markdown("**📊 Normalized Features:**")
        st.dataframe(
            normalized_df.describe().round(3),
            use_container_width=True
        )
    
    # Correlation Analysis
    st.subheader("🔗 Enhanced Correlation Analysis")
    
    # Create correlation matrix with normalized features
    correlation_data = pd.concat([
        distances_df[['total_population', 'median_income', 'car_ownership_rate', 'distance_to_nearest_stop_km']],
        normalized_df
    ], axis=1)
    
    correlation_matrix = correlation_data.corr()
    
    # Create correlation heatmap
    fig_corr = px.imshow(
        correlation_matrix,
        title='Correlation Matrix (Original + Normalized Features)',
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # K-Means Clustering Section
    st.subheader("🎯 STEP 4: K-Means Clustering Analysis")
    
    # Display clustering parameters from sidebar
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Clusters", n_clusters)
    
    with col2:
        st.markdown("**Clustering Features:**")
        for feature in feature_columns:
            st.write(f"• {feature}")
    
    # Perform clustering
    with st.spinner(f"🔄 Performing K-means clustering with {n_clusters} clusters..."):
        cluster_labels, kmeans = perform_kmeans_clustering(normalized_df, n_clusters)
        distances_with_clusters, cluster_stats, cluster_to_equity = analyze_clusters(distances_df, cluster_labels)
    
    st.success(f"✅ Clustering complete! Identified {n_clusters} distinct neighborhood types.")
    
    # Cluster Analysis Results
    st.subheader("📊 Cluster Analysis Results")
    
    # Cluster statistics
    st.markdown("**🏘️ Cluster Characteristics:**")
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Cluster visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Distance vs Income by cluster
        fig_cluster_scatter = px.scatter(
            distances_with_clusters,
            x='distance_to_nearest_stop_km',
            y='median_income',
            color='equity_label',
            title='Distance vs Income by Cluster',
            labels={'distance_to_nearest_stop_km': 'Distance (km)', 'median_income': 'Median Income ($)'},
            hover_data=['tract_name', 'total_population']
        )
        st.plotly_chart(fig_cluster_scatter, use_container_width=True)
    
    with col2:
        # Cluster distribution
        cluster_counts = distances_with_clusters['equity_label'].value_counts()
        fig_cluster_dist = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title='Distribution of Neighborhood Types',
            labels={'x': 'Equity Level', 'y': 'Number of Tracts'}
        )
        st.plotly_chart(fig_cluster_dist, use_container_width=True)
    
    # Interactive cluster map
    st.subheader("🗺️ Interactive Cluster Map")
    
    # Create map with cluster coloring
    if 'stop_lat' in gtfs_df.columns:
        center_lat = (gtfs_df['stop_lat'].mean() + distances_with_clusters['latitude'].mean()) / 2
        center_lon = (gtfs_df['stop_lon'].mean() + distances_with_clusters['longitude'].mean()) / 2
        
        cluster_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Color scheme for clusters
        cluster_colors = {
            "🚨 Severely Underserved": "red",
            "⚠️ Moderately Underserved": "orange", 
            "✅ Adequately Served": "yellow",
            "🌟 Well Served": "green"
        }
        
        # Add transit stops
        for idx, row in gtfs_df.head(30).iterrows():
            folium.Marker(
                [row['stop_lat'], row['stop_lon']],
                popup=f"<b>{row['stop_name']}</b><br>ID: {row['stop_id']}",
                icon=folium.Icon(color='blue', icon='train')
            ).add_to(cluster_map)
        
        # Add census tracts with cluster coloring
        for idx, row in distances_with_clusters.head(40).iterrows():
            color = cluster_colors.get(row['equity_label'], 'gray')
            
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=12,
                popup=f"<b>{row['tract_name']}</b><br>Equity: {row['equity_label']}<br>Population: {row['total_population']:,}<br>Income: ${row['median_income']:,}<br>Distance: {row['distance_to_nearest_stop_km']:.2f} km",
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(cluster_map)
        
        st_folium(cluster_map, width=800, height=600)
    
    # New Stop Proposals Section
    st.subheader("🚏 STEP 6: Propose New Stops")
    
    # Ensure we have equity scores
    if 'transit_equity_score' not in distances_df.columns:
        st.warning("⚠️ Please calculate equity scores first in the Transit Equity Analysis tab.")
        return
    
    # Display proposal parameters from sidebar
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Equity Threshold", f"{equity_threshold:.2f}")
    
    with col2:
        st.metric("Max Proposals", max_proposals)
    
    with col3:
        st.markdown("**Proposal Criteria:**")
        st.write(f"• Equity score < {equity_threshold}")
        st.write(f"• Top {max_proposals} priority areas")
    
    # Generate proposals
    with st.spinner(f"🔄 Generating new stop proposals for areas with equity score < {equity_threshold}..."):
        proposed_stops_df, impact_analysis_df = propose_new_stops(distances_df, gtfs_df, equity_threshold, max_proposals)
    
    if len(proposed_stops_df) > 0:
        st.success(f"✅ Generated {len(proposed_stops_df)} new stop proposals!")
        
        # Proposed stops table
        st.markdown("**📋 Proposed New Transit Stops:**")
        st.dataframe(
            proposed_stops_df[['proposed_stop_name', 'tract_name', 'current_equity_score', 'population_served', 'median_income', 'current_distance']].round(3),
            use_container_width=True
        )
        
        # Impact analysis
        st.markdown("**📊 Impact Analysis:**")
        st.dataframe(
            impact_analysis_df[['proposed_stop', 'tract_name', 'current_equity_score', 'new_equity_score', 'equity_improvement_pct', 'tracts_improved', 'total_population_impacted']].round(3),
            use_container_width=True
        )
        
        # Impact visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity improvement chart
            fig_improvement = px.bar(
                impact_analysis_df,
                x='proposed_stop',
                y='equity_improvement_pct',
                title='Equity Score Improvement (%)',
                labels={'proposed_stop': 'Proposed Stop', 'equity_improvement_pct': 'Improvement (%)'}
            )
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        with col2:
            # Population impact chart
            fig_pop_impact = px.bar(
                impact_analysis_df,
                x='proposed_stop',
                y='total_population_impacted',
                title='Total Population Impacted',
                labels={'proposed_stop': 'Proposed Stop', 'total_population_impacted': 'Population Impacted'}
            )
            st.plotly_chart(fig_pop_impact, use_container_width=True)
        
        # Proposed stops map
        st.subheader("🗺️ Proposed New Stops Map")
        
        with st.spinner("🔄 Creating proposed stops map..."):
            proposed_map = create_proposed_stops_map(distances_df, gtfs_df, proposed_stops_df)
        
        st_folium(proposed_map, width=800, height=600)
        
        # Summary statistics
        st.subheader("📈 Proposal Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_improvement = impact_analysis_df['equity_improvement_pct'].mean()
            st.metric("📊 Avg Equity Improvement", f"{avg_improvement:.1f}%")
        
        with col2:
            total_pop_impacted = impact_analysis_df['total_population_impacted'].sum()
            st.metric("👥 Total Population Impacted", f"{total_pop_impacted:,}")
        
        with col3:
            total_tracts_improved = impact_analysis_df['tracts_improved'].sum()
            st.metric("🏘️ Total Tracts Improved", total_tracts_improved)
        
        with col4:
            avg_distance_improvement = impact_analysis_df['avg_distance_improvement'].mean()
            st.metric("📏 Avg Distance Improvement", f"{avg_distance_improvement:.2f} km")
        
        # Policy recommendations
        st.subheader("📋 New Stop Recommendations")
        
        st.markdown(f"""
        ### 🎯 Strategic Recommendations:
        
        **Proposed Stop Locations:**
        - **{len(proposed_stops_df)} new stops** recommended for areas with equity scores below {equity_threshold}
        - **Average equity improvement:** {avg_improvement:.1f}% across all proposals
        - **Total population impacted:** {total_pop_impacted:,} residents
        
        **Implementation Priority:**
        1. **High Impact Stops:** Focus on stops with >{avg_improvement:.1f}% equity improvement
        2. **Population Density:** Prioritize stops serving >{total_pop_impacted/len(proposed_stops_df):,.0f} residents
        3. **Geographic Distribution:** Ensure equitable coverage across underserved areas
        
        **Expected Outcomes:**
        - **Reduced transit deserts** in {len(proposed_stops_df)} priority areas
        - **Improved accessibility** for {total_pop_impacted:,} residents
        - **Enhanced equity scores** across {total_tracts_improved} census tracts
        """)
        
        # Download proposals
        st.subheader("📥 Download Proposal Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            proposals_csv = proposed_stops_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Proposed Stops (CSV)",
                data=proposals_csv,
                file_name="proposed_transit_stops.csv",
                mime="text/csv"
            )
        
        with col2:
            impact_csv = impact_analysis_df.to_csv(index=False)
            st.download_button(
                label="📈 Download Impact Analysis (CSV)",
                data=impact_csv,
                file_name="stop_proposal_impact.csv",
                mime="text/csv"
            )
    
    else:
        st.warning(f"⚠️ No new stop proposals found for equity threshold {equity_threshold}. Try lowering the threshold or check equity score calculations.")
    
    # Detailed cluster analysis
    st.subheader("🔍 Detailed Cluster Analysis")
    
    # Show tracts by cluster
    for equity_label in cluster_colors.keys():
        cluster_tracts = distances_with_clusters[distances_with_clusters['equity_label'] == equity_label]
        
        if len(cluster_tracts) > 0:
            st.markdown(f"**{equity_label} ({len(cluster_tracts)} tracts):**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Distance", f"{cluster_tracts['distance_to_nearest_stop_km'].mean():.2f} km")
            
            with col2:
                st.metric("Avg Income", f"${cluster_tracts['median_income'].mean():,.0f}")
            
            with col3:
                st.metric("Total Population", f"{cluster_tracts['total_population'].sum():,}")
            
            with col4:
                st.metric("Avg Car Ownership", f"{cluster_tracts['car_ownership_rate'].mean():.1%}")
            
            # Show top tracts in this cluster
            st.dataframe(
                cluster_tracts[['tract_name', 'distance_to_nearest_stop_km', 'median_income', 'total_population', 'nearest_stop_name']].round(2),
                use_container_width=True
            )
    
    # Machine Learning Insights
    st.subheader("🤖 Machine Learning Insights")
    
    st.markdown("""
    ### 📈 Key Findings from Clustering:
    
    **Cluster Analysis Reveals:**
    - **Distinct neighborhood types** based on transit access and demographics
    - **Equity patterns** in transit accessibility across income levels
    - **Transit desert identification** through data-driven clustering
    
    **Policy Implications:**
    - **Targeted interventions** for each cluster type
    - **Resource allocation** based on equity levels
    - **Transit planning** informed by demographic patterns
    """)
    
    # Download processed data
    st.subheader("📥 Download Processed Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download cluster results
        cluster_csv = distances_with_clusters.to_csv(index=False)
        st.download_button(
            label="📊 Download Cluster Results (CSV)",
            data=cluster_csv,
            file_name="transit_equity_clusters.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download cluster statistics
        stats_csv = cluster_stats.to_csv()
        st.download_button(
            label="📈 Download Cluster Statistics (CSV)",
            data=stats_csv,
            file_name="cluster_statistics.csv",
            mime="text/csv"
        )
    
    with col3:
        # PDF Report Generation
        st.markdown("**📄 Generate PDF Report**")
        if st.button("📋 Create Report", help="Generate a comprehensive PDF report with all analysis results"):
            with st.spinner("🔄 Generating PDF report..."):
                try:
                    # Get proposal data if available
                    proposed_stops_df = None
                    impact_analysis_df = None
                    
                    # Try to get proposal data
                    if 'transit_equity_score' in distances_with_clusters.columns:
                        try:
                            proposed_stops_df, impact_analysis_df = propose_new_stops(distances_with_clusters, gtfs_df, equity_threshold, max_proposals)
                        except:
                            pass  # Proposals not available
                    
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(distances_df, gtfs_df, cluster_stats, proposed_stops_df, impact_analysis_df)
                    
                    st.success("✅ PDF report generated successfully!")
                    
                    # Download button for PDF
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"transit_equity_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        help="Download the comprehensive PDF report with all analysis results"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF report: {str(e)}")

if __name__ == "__main__":
    main()
