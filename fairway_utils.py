import math
import io
import pandas as pd
import numpy as np
import folium
import streamlit as st
from streamlit_folium import st_folium


# ---------- UI Helpers ----------

def inject_styles():
    st.markdown(
        """
        <style>
        :root {
          --fairway-primary: #1f77b4; /* blue */
          --fairway-accent: #2ecc71;  /* green */
          --fairway-bg1: #e8f1fb;
          --fairway-bg2: #eafaf1;
        }
        .fairway-hero {
          background: linear-gradient(135deg, var(--fairway-bg1), var(--fairway-bg2));
          padding: 3rem 2rem; border-radius: 16px; position: relative; overflow: hidden;
          animation: fadeIn 0.8s ease-in-out;
        }
        .fairway-title { font-size: 3rem; font-weight: 800; color: var(--fairway-primary); margin: 0; }
        .fairway-subtitle { font-size: 1.25rem; color: #2c3e50; margin-top: 0.5rem; }
        .fairway-slogan { color: var(--fairway-accent); font-weight: 700; }
        .fairway-nav { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; }
        .fairway-nav a { text-decoration: none; padding: 0.5rem 0.9rem; border-radius: 8px; color: #1b1f24; background: #f6f8fa; border: 1px solid #e1e4e8; }
        .fairway-nav a.active { background: var(--fairway-primary); color: white; border-color: var(--fairway-primary); }
        .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid var(--fairway-primary); }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(active: str):
    cols = st.columns([1, 1, 1, 1, 1])
    items = [
        ("Home", "app.py"),
        ("Data Upload", "pages/1_📁_Data_Upload.py"),
        ("Analysis", "pages/2_📊_Analysis.py"),
        ("Proposals", "pages/3_🚏_Proposals.py"),
        ("About", "pages/4_ℹ️_About.py"),
    ]
    for i, (label, target) in enumerate(items):
        with cols[i]:
            cls = "active" if label == active else ""
            st.markdown(f'<div class="fairway-nav"><a class="{cls}" href="#" onclick="window.parent.postMessage({{type: \'streamlit_navigation\', page: \'{target}\'}}, \'*\')">{label}</a></div>', unsafe_allow_html=True)

    # Small JS to call st.switch_page via query param workaround
    st.markdown(
        """
        <script>
        window.addEventListener('message', (event) => {
          const data = event.data || {};
          if (data.type === 'streamlit_navigation' && data.page) {
            const pyMsg = {isNavigation: true, page: data.page};
            const el = document.getElementById('nav-target');
            if (el) { el.innerText = JSON.stringify(pyMsg); }
          }
        });
        </script>
        <div id="nav-target" style="display:none"></div>
        """,
        unsafe_allow_html=True,
    )
    # Read injected message
    nav_msg = st.empty()
    try:
        raw = st.session_state.get("_nav_raw", None)
        txt = nav_msg.text("", help="internal")
    except Exception:
        pass
    # Use experimental switch if available
    try:
        import streamlit
        if hasattr(st, "switch_page"):
            placeholder = st.empty()
            key = "_nav_text_holder"
            val = st.session_state.get(key, "")
            # Provide a button-based fallback
    except Exception:
        pass


# ---------- Data + Analysis Utilities ----------

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


def calculate_distances_to_transit(gtfs_df, census_df):
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
                nearest_stop = stop_row.get('stop_name', stop_row.get('stop_id', ''))
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
            'nearest_stop_name': nearest_stop,
        })
    return pd.DataFrame(distances)


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def normalize_features(distances_df):
    feature_columns = ['median_income', 'car_ownership_rate', 'distance_to_nearest_stop_km']
    if 'population_density' not in distances_df.columns:
        distances_df['population_density'] = distances_df['total_population'] / distances_df['area_sq_km']
    feature_columns.append('population_density')
    feature_matrix = distances_df[feature_columns].copy().fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    normalized_df = pd.DataFrame(
        normalized_features,
        columns=[f'{col}_normalized' for col in feature_columns],
        index=distances_df.index,
    )
    return normalized_df, scaler, feature_columns


def perform_kmeans_clustering(normalized_df, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_df)
    return cluster_labels, kmeans


def analyze_clusters(distances_df, cluster_labels):
    distances_df_with_clusters = distances_df.copy()
    distances_df_with_clusters['cluster'] = cluster_labels
    cluster_stats = distances_df_with_clusters.groupby('cluster').agg({
        'distance_to_nearest_stop_km': ['mean', 'std'],
        'median_income': ['mean', 'std'],
        'total_population': ['mean', 'sum'],
        'car_ownership_rate': 'mean',
        'population_density': 'mean'
    }).round(3)
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
    cluster_stats = cluster_stats.sort_values('distance_to_nearest_stop_km_mean')
    base_equity_labels = [
        "🚨 Severely Underserved",
        "⚠️ Moderately Underserved",
        "✅ Adequately Served",
        "🌟 Well Served",
    ]
    equity_labels = {}
    for i in range(len(cluster_stats)):
        equity_labels[i] = base_equity_labels[i] if i < len(base_equity_labels) else f"Cluster {i+1}"
    cluster_to_equity = {cluster_num: equity_labels[i] for i, cluster_num in enumerate(cluster_stats.index)}
    distances_df_with_clusters['equity_label'] = distances_df_with_clusters['cluster'].map(cluster_to_equity)
    return distances_df_with_clusters, cluster_stats, cluster_to_equity


def calculate_transit_equity_score(distances_df, weights=None):
    if weights is None:
        weights = {'population_density': 0.4, 'distance': 0.4, 'income': 0.2}
    if 'population_density' not in distances_df.columns:
        distances_df['population_density'] = distances_df['total_population'] / distances_df['area_sq_km']
    pop_density_norm = (distances_df['population_density'] - distances_df['population_density'].min()) / \
                      (distances_df['population_density'].max() - distances_df['population_density'].min() + 1e-9)
    distance_component = 1 / (distances_df['distance_to_nearest_stop_km'] + 0.1)
    distance_norm = (distance_component - distance_component.min()) / (distance_component.max() - distance_component.min() + 1e-9)
    income_component = 1 / (distances_df['median_income'] + 1000)
    income_norm = (income_component - income_component.min()) / (income_component.max() - income_component.min() + 1e-9)
    equity_score = (weights['population_density'] * pop_density_norm +
                   weights['distance'] * distance_norm +
                   weights['income'] * income_norm)
    equity_score_norm = (equity_score - equity_score.min()) / (equity_score.max() - equity_score.min() + 1e-9)
    return equity_score_norm


def create_equity_choropleth_map(distances_df, gtfs_df):
    center_lat = (gtfs_df['stop_lat'].mean() + distances_df['latitude'].mean()) / 2
    center_lon = (gtfs_df['stop_lon'].mean() + distances_df['longitude'].mean()) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    for _, row in gtfs_df.head(200).iterrows():
        folium.Marker([row['stop_lat'], row['stop_lon']], icon=folium.Icon(color='blue', icon='train')).add_to(m)
    for _, row in distances_df.iterrows():
        score = row['transit_equity_score']
        color = 'red' if score < 0.3 else 'orange' if score < 0.6 else 'yellow' if score < 0.8 else 'green'
        folium.CircleMarker([row['latitude'], row['longitude']], radius=10, color=color, fill=True, fillOpacity=0.7,
                            popup=f"{row['tract_name']} — Equity: {score:.3f}").add_to(m)
    return m


def propose_new_stops(distances_df, gtfs_df, equity_threshold=0.3, max_proposals=5):
    low_equity_tracts = distances_df[distances_df['transit_equity_score'] < equity_threshold].copy()
    if len(low_equity_tracts) == 0:
        return pd.DataFrame(), pd.DataFrame()
    low_equity_tracts = low_equity_tracts.nsmallest(max_proposals, 'transit_equity_score')
    proposed_stops = []
    impact_analysis = []
    for idx, tract in low_equity_tracts.iterrows():
        proposed_lat = tract['latitude']
        proposed_lon = tract['longitude']
        new_distances = []
        for _, other_tract in distances_df.iterrows():
            d_prop = haversine_distance(other_tract['latitude'], other_tract['longitude'], proposed_lat, proposed_lon)
            curr = other_tract['distance_to_nearest_stop_km']
            new_distances.append(min(curr, d_prop))
        distances_with_new = distances_df.copy()
        distances_with_new['distance_to_nearest_stop_km'] = new_distances
        new_equity_scores = calculate_transit_equity_score(distances_with_new)
        equity_improvement = float(new_equity_scores.loc[idx] - distances_df['transit_equity_score'].loc[idx])
        proposed_stops.append({
            'proposed_stop_id': f"PROPOSED_{len(proposed_stops)+1:03d}",
            'proposed_stop_name': f"New Stop - {tract['tract_name']}",
            'latitude': proposed_lat,
            'longitude': proposed_lon,
            'tract_name': tract['tract_name'],
            'current_equity_score': float(tract['transit_equity_score']),
            'population_served': int(tract['total_population']),
            'median_income': float(tract['median_income']),
        })
        impact_analysis.append({
            'proposed_stop': f"New Stop - {tract['tract_name']}",
            'tract_name': tract['tract_name'],
            'current_equity_score': float(tract['transit_equity_score']),
            'new_equity_score': float(new_equity_scores.loc[idx]),
            'equity_improvement_pct': float((equity_improvement / (tract['transit_equity_score'] + 1e-9)) * 100.0),
        })
    return pd.DataFrame(proposed_stops), pd.DataFrame(impact_analysis)


def compute_underserved_candidates(distances_df, weights, bottom_quantile: float = 0.3):
    distances_with_equity = distances_df.copy()
    distances_with_equity['transit_equity_score'] = calculate_transit_equity_score(distances_with_equity, weights)
    threshold = distances_with_equity['transit_equity_score'].quantile(bottom_quantile)
    candidates_df = distances_with_equity[distances_with_equity['transit_equity_score'] <= threshold].copy()
    return distances_with_equity, candidates_df


def compute_tract_weights(distances_df: pd.DataFrame, strategy: str, income_multiplier: float = 1.8) -> np.ndarray:
    population = distances_df['total_population'].to_numpy().astype(float)
    if 'population_density' not in distances_df.columns:
        density = (distances_df['total_population'] / distances_df.get('area_sq_km', 1.0)).to_numpy().astype(float)
    else:
        density = distances_df['population_density'].to_numpy().astype(float)
    income = distances_df['median_income'].to_numpy().astype(float)
    def safe_norm(arr):
        arr = arr.astype(float)
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        if max_v - min_v < 1e-9:
            return np.ones_like(arr)
        return (arr - min_v) / (max_v - min_v)
    density_norm = safe_norm(density)
    income_inv = 1.0 / (income + 1000.0)
    income_norm = safe_norm(income_inv)
    if strategy == 'Equity-first':
        factor = 1.0 + (income_multiplier - 1.0) * income_norm
        weights_arr = population * factor
    elif strategy == 'Density-first':
        factor = 0.5 + 1.5 * density_norm
        weights_arr = population * factor
    else:
        factor_income = 1.0 + (income_multiplier - 1.0) * income_norm
        factor_density = 0.5 + 1.5 * density_norm
        factor = 0.5 * factor_income + 0.5 * factor_density
        weights_arr = population * factor
    weights_arr = np.where(weights_arr < 0, 0.0, weights_arr)
    if float(weights_arr.sum()) <= 0:
        weights_arr = population.copy()
    return weights_arr


def greedy_optimize_new_stops(distances_df, candidates_df, k: int, weights, strategy: str = 'Balanced', income_multiplier: float = 1.8):
    current_min_distance = distances_df['distance_to_nearest_stop_km'].to_numpy().astype(float)
    tract_latitudes = distances_df['latitude'].to_numpy()
    tract_longitudes = distances_df['longitude'].to_numpy()
    tract_weights = compute_tract_weights(distances_df, strategy=strategy, income_multiplier=income_multiplier)
    selected = []
    used_candidate_ids = set()
    for _ in range(max(0, k)):
        best_idx = None; best_weighted_avg = float('inf'); best_new_min = None; best_stats = None
        for idx, cand in candidates_df.iterrows():
            if idx in used_candidate_ids:
                continue
            cand_lat = float(cand['latitude']); cand_lon = float(cand['longitude'])
            cand_distances = np.array([
                haversine_distance(float(tract_latitudes[i]), float(tract_longitudes[i]), cand_lat, cand_lon)
                for i in range(len(tract_latitudes))
            ])
            new_min = np.minimum(current_min_distance, cand_distances)
            weighted_avg = float(np.average(new_min, weights=tract_weights)) if tract_weights.sum() > 0 else float(new_min.mean())
            improvement = current_min_distance - new_min
            impacted_mask = improvement > 1e-9
            impacted_pop = float(tract_weights[impacted_mask].sum())
            avg_distance_reduction = float(improvement[impacted_mask].mean()) if impacted_mask.any() else 0.0
            tmp_df = distances_df.copy(); tmp_df['distance_to_nearest_stop_km'] = new_min
            new_equity = calculate_transit_equity_score(tmp_df, weights)
            curr_equity = calculate_transit_equity_score(distances_df, weights)
            equity_improvement_pct = float(((new_equity.mean() - curr_equity.mean()) / (curr_equity.mean() + 1e-9)) * 100)
            if weighted_avg < best_weighted_avg:
                best_weighted_avg = weighted_avg; best_idx = idx; best_new_min = new_min
                best_stats = {
                    'proposed_stop_id': f"OPT_{len(selected)+1:03d}",
                    'proposed_stop_name': f"Optimized Stop {len(selected)+1}",
                    'latitude': cand_lat,
                    'longitude': cand_lon,
                    'tract_name': cand.get('tract_name', cand.get('tract_id', str(best_idx))),
                    'population_served': int(round(impacted_pop)),
                    'avg_distance_reduction_km': avg_distance_reduction,
                    'equity_improvement_pct': equity_improvement_pct,
                }
        if best_idx is None:
            break
        used_candidate_ids.add(best_idx)
        current_min_distance = best_new_min
        selected.append(best_stats)
    return selected, current_min_distance


def compute_before_after_metrics(distances_df, before_distances, after_distances, weights):
    before_df = distances_df.copy(); after_df = distances_df.copy()
    before_df['distance_to_nearest_stop_km'] = before_distances
    after_df['distance_to_nearest_stop_km'] = after_distances
    before_df['transit_equity_score'] = calculate_transit_equity_score(before_df, weights)
    after_df['transit_equity_score'] = calculate_transit_equity_score(after_df, weights)
    def pct_pop_within_threshold(df, threshold_km=0.5):
        within = df['distance_to_nearest_stop_km'] <= threshold_km
        return float(df.loc[within, 'total_population'].sum() / (df['total_population'].sum() + 1e-9) * 100.0)
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


def create_proposed_stops_map(distances_df, gtfs_df, proposed_stops_df):
    center_lat = (gtfs_df['stop_lat'].mean() + distances_df['latitude'].mean()) / 2
    center_lon = (gtfs_df['stop_lon'].mean() + distances_df['longitude'].mean()) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    for _, row in gtfs_df.head(200).iterrows():
        folium.Marker([row['stop_lat'], row['stop_lon']], icon=folium.Icon(color='blue', icon='train')).add_to(m)
    for _, row in proposed_stops_df.iterrows():
        folium.Marker([row['latitude'], row['longitude']], icon=folium.Icon(color='red', icon='star'),
                      popup=f"{row['proposed_stop_name']} — {row['tract_name']}").add_to(m)
    return m


def create_before_after_maps(before_df, after_df, gtfs_df, proposed_stops):
    center_lat = (gtfs_df['stop_lat'].mean() + before_df['latitude'].mean()) / 2
    center_lon = (gtfs_df['stop_lon'].mean() + before_df['longitude'].mean()) / 2
    def color_from_equity(score: float) -> str:
        if score < 0.3: return 'red'
        if score < 0.6: return 'orange'
        if score < 0.8: return 'yellow'
        return 'green'
    m_before = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    for _, row in gtfs_df.iterrows():
        folium.Marker([row['stop_lat'], row['stop_lon']], icon=folium.Icon(color='blue', icon='train')).add_to(m_before)
    for _, row in before_df.iterrows():
        color = color_from_equity(float(row['transit_equity_score']))
        folium.CircleMarker([row['latitude'], row['longitude']], radius=10, color=color, fill=True, fillOpacity=0.7).add_to(m_before)
    m_after = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    for _, row in gtfs_df.iterrows():
        folium.Marker([row['stop_lat'], row['stop_lon']], icon=folium.Icon(color='blue', icon='train')).add_to(m_after)
    for prop in proposed_stops:
        folium.Marker([prop['latitude'], prop['longitude']], icon=folium.Icon(color='red', icon='star')).add_to(m_after)
    for _, row in after_df.iterrows():
        color = color_from_equity(float(row['transit_equity_score']))
        folium.CircleMarker([row['latitude'], row['longitude']], radius=10, color=color, fill=True, fillOpacity=0.7).add_to(m_after)
    return m_before, m_after


def simulate_routes_shortest_path(tracts_df: pd.DataFrame, gtfs_df: pd.DataFrame, proposed_stops: list, num_cases: int = 6) -> pd.DataFrame:
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
        records.append({'tract_name': tract_row['tract_name'], 'before_time_min': before_time, 'after_time_min': after_time, 'improvement_pct': improvement_pct})
    return pd.DataFrame(records).sort_values('improvement_pct', ascending=False)


def draw_routes_on_map(m: folium.Map, routes_df: pd.DataFrame, tracts_df: pd.DataFrame, gtfs_df: pd.DataFrame, proposed_stops: list | None, color: str):
    gtfs_coords = [(float(r['stop_lat']), float(r['stop_lon'])) for _, r in gtfs_df.iterrows()]
    proposed_coords = [(float(s['latitude']), float(s['longitude'])) for s in (proposed_stops or [])]
    def nearest_coord(lat: float, lon: float):
        choices = gtfs_coords + proposed_coords if proposed_stops else gtfs_coords
        if not choices:
            return lat, lon
        best = min(choices, key=lambda c: haversine_distance(lat, lon, c[0], c[1]))
        return best
    for _, case in routes_df.iterrows():
        tract = tracts_df[tracts_df['tract_name'] == case['tract_name']].iloc[0]
        lat, lon = float(tract['latitude']), float(tract['longitude'])
        end_lat, end_lon = nearest_coord(lat, lon)
        folium.PolyLine([[lat, lon], [end_lat, end_lon]], color=color, weight=2, opacity=0.6).add_to(m)
    return m


def load_real_data():
    try:
        gtfs_df = pd.read_csv("data/real_gtfs_stops_processed.csv")
        census_df = pd.read_csv("data/real_census_tracts_processed.csv")
        return gtfs_df, census_df
    except Exception:
        return None, None


def validate_data(gtfs_df, census_df):
    required_gtfs_cols = ['stop_id', 'stop_lat', 'stop_lon']
    required_census_cols = ['tract_id', 'latitude', 'longitude', 'total_population', 'median_income']
    gtfs_valid = all(col in gtfs_df.columns for col in required_gtfs_cols)
    census_valid = all(col in census_df.columns for col in required_census_cols)
    return gtfs_valid and census_valid


