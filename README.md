# FairWay — Rethinking Routes

FairWay is a multi-page Streamlit application that analyzes transit equity and proposes optimized stop locations using real GTFS stops and Census tract data. It provides interactive maps, clustering, an equity score, greedy facility-location optimization with equity/density weighting, and a simple travel-time simulation for before/after comparisons.

## Quick start

1. Create and activate a Python environment (3.10–3.12 recommended).
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the browser at http://localhost:8501 (or the printed port).

## Data
- GTFS stops: `data/real_gtfs_stops_processed.csv`
- Census tracts: `data/real_census_tracts_processed.csv`

You can also upload your own CSVs on the Data Upload page. Required columns:
- Transit (GTFS stops): `stop_id`, `stop_lat`, `stop_lon`
- Census: `tract_id`, `latitude`, `longitude`, `total_population`, `median_income` (optional: `population_density`, `car_ownership_rate`, `area_sq_km`, `poverty_rate`)

## Application pages
- Home: overview, quick navigation, and links to analysis.
- Data Upload: upload/validate GTFS and Census CSVs; preview first rows.
- Analysis:
  - Equity choropleth map
  - Feature normalization and correlation heatmap
  - K-Means clustering (adjustable k)
  - Before/After optimization with weighting strategies (Equity-first, Density-first, Balanced), side-by-side maps, key metrics, and route-time simulation
- Proposals: proposed stops table and map
- About: mission, team, credits

## Methods
- Distance to transit: Haversine to nearest stop (km)
- Equity score: normalized combination of population density, inverse distance, and inverse income with adjustable weights
- Clustering: K-Means on normalized features
- Optimization: greedy facility-location heuristic that selects k candidate tract centroids to minimize weighted average distance
- Travel-time simulation: walking time to nearest stop plus a fixed in-vehicle component to a hub (illustrative)

## Configuration
- Streamlit options can be passed via CLI or `~/.streamlit/config.toml`.
- For Streamlit Community Cloud, ensure Python version is compatible and `requirements.txt` is present.

## Repository structure
```
app.py                     # Home page entry
fairway_utils.py           # Core computation + UI style helpers
pages/                     # Multi-page app files
  1_Data_Upload.py
  2_Analysis.py
  3_Proposals.py
  4_About.py
data/                      # Sample/processed datasets
requirements.txt           # Python dependencies
README.md                  # This file
```

## Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app pointing to `app.py` on the `main` branch.
3. Set secrets/environment variables only if needed (none required by default).
4. Deploy.

## License
MIT
