import pandas as pd
import requests
import zipfile
import io
import os
from pathlib import Path

def download_and_load_gtfs_data():
    """
    Download and load GTFS transit data from Kaggle
    """
    print("🔍 STEP 0: Data Discovery & Ingestion")
    print("=" * 50)
    
    # GTFS Data Sources
    gtfs_sources = {
        "kaggle_gtfs": "https://www.kaggle.com/datasets/harrywang/gtfs-public-transportation-data",
        "mta_nyc": "https://transitfeeds.com/p/mta/79",
        "chicago_cta": "https://transitfeeds.com/p/chicago-transit-authority/165"
    }
    
    print("📊 GTFS Transit Data Sources:")
    for name, url in gtfs_sources.items():
        print(f"  - {name}: {url}")
    
    # For now, let's create a sample GTFS stops dataset
    # In a real scenario, you would download from these sources
    print("\n📥 Creating sample GTFS stops data...")
    
    # Sample GTFS stops data (you can replace this with actual downloaded data)
    sample_stops = {
        'stop_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
        'stop_name': ['Central Station', 'Downtown Hub', 'University Stop', 'Airport Terminal', 'Shopping Center'],
        'stop_lat': [40.7128, 40.7589, 40.7505, 40.6413, 40.7505],
        'stop_lon': [-74.0060, -73.9851, -73.9934, -73.7781, -73.9934],
        'stop_desc': ['Main transit hub', 'Downtown connection', 'University area', 'Airport access', 'Shopping district']
    }
    
    gtfs_stops_df = pd.DataFrame(sample_stops)
    
    print("✅ GTFS Stops Data Schema:")
    print(gtfs_stops_df.info())
    print("\n📋 First 5 rows of GTFS stops:")
    print(gtfs_stops_df.head())
    
    return gtfs_stops_df

def download_and_load_census_data():
    """
    Download and load US Census demographic data
    """
    print("\n" + "=" * 50)
    print("📊 US Census Demographic Data Sources:")
    
    census_sources = {
        "kaggle_census": "https://www.kaggle.com/datasets/muonneutrino/us-census-demographic-data",
        "census_gov": "https://data.census.gov/",
        "nhgis": "https://www.nhgis.org/",
        "census_geojson": "https://github.com/loganpowell/census-geojson"
    }
    
    for name, url in census_sources.items():
        print(f"  - {name}: {url}")
    
    print("\n📥 Creating sample Census demographic data...")
    
    # Sample Census tract data (you can replace this with actual downloaded data)
    sample_census = {
        'tract_id': ['36061000100', '36061000200', '36061000300', '36061000400', '36061000500'],
        'tract_name': ['Downtown Core', 'University District', 'Airport Area', 'Suburban West', 'Industrial Zone'],
        'latitude': [40.7128, 40.7505, 40.6413, 40.7589, 40.7505],
        'longitude': [-74.0060, -73.9934, -73.7781, -73.9851, -73.9934],
        'total_population': [8500, 12000, 3200, 8900, 2100],
        'median_income': [45000, 65000, 38000, 72000, 35000],
        'car_ownership_rate': [0.35, 0.25, 0.45, 0.15, 0.60],
        'area_sq_km': [2.1, 3.5, 8.2, 4.8, 12.3],
        'poverty_rate': [0.18, 0.12, 0.22, 0.08, 0.25]
    }
    
    census_df = pd.DataFrame(sample_census)
    
    # Calculate population density
    census_df['population_density'] = census_df['total_population'] / census_df['area_sq_km']
    
    print("✅ Census Data Schema:")
    print(census_df.info())
    print("\n📋 First 5 rows of Census data:")
    print(census_df.head())
    
    return census_df

def analyze_join_potential(gtfs_df, census_df):
    """
    Analyze the potential for joining GTFS and Census data
    """
    print("\n" + "=" * 50)
    print("🔗 Data Join Analysis:")
    
    print(f"\n📍 GTFS Stops Data:")
    print(f"  - Shape: {gtfs_df.shape}")
    print(f"  - Columns: {list(gtfs_df.columns)}")
    print(f"  - Has lat/lon: {'stop_lat' in gtfs_df.columns and 'stop_lon' in gtfs_df.columns}")
    
    print(f"\n🏘️ Census Data:")
    print(f"  - Shape: {census_df.shape}")
    print(f"  - Columns: {list(census_df.columns)}")
    print(f"  - Has lat/lon: {'latitude' in census_df.columns and 'longitude' in census_df.columns}")
    
    print(f"\n🎯 Join Strategy:")
    print(f"  - Both datasets have lat/lon coordinates ✓")
    print(f"  - Can use spatial joins or distance calculations")
    print(f"  - GTFS stops: {len(gtfs_df)} locations")
    print(f"  - Census tracts: {len(census_df)} areas")
    
    # Check for required columns for transit equity analysis
    required_census_cols = ['total_population', 'median_income', 'car_ownership_rate']
    required_gtfs_cols = ['stop_lat', 'stop_lon']
    
    missing_census = [col for col in required_census_cols if col not in census_df.columns]
    missing_gtfs = [col for col in required_gtfs_cols if col not in gtfs_df.columns]
    
    if missing_census:
        print(f"  ⚠️  Missing Census columns: {missing_census}")
    else:
        print(f"  ✅ All required Census columns present")
        
    if missing_gtfs:
        print(f"  ⚠️  Missing GTFS columns: {missing_gtfs}")
    else:
        print(f"  ✅ All required GTFS columns present")

def save_sample_data(gtfs_df, census_df):
    """
    Save the sample data to CSV files for use in the Streamlit app
    """
    print("\n" + "=" * 50)
    print("💾 Saving sample data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save datasets
    gtfs_df.to_csv(data_dir / "sample_gtfs_stops.csv", index=False)
    census_df.to_csv(data_dir / "sample_census_data.csv", index=False)
    
    print(f"✅ Saved GTFS data to: {data_dir / 'sample_gtfs_stops.csv'}")
    print(f"✅ Saved Census data to: {data_dir / 'sample_census_data.csv'}")
    
    return data_dir

def main():
    """
    Main function to run the data discovery and ingestion process
    """
    print("🚌 TransitFair: Data Discovery & Ingestion")
    print("=" * 60)
    
    # Load GTFS data
    gtfs_df = download_and_load_gtfs_data()
    
    # Load Census data
    census_df = download_and_load_census_data()
    
    # Analyze join potential
    analyze_join_potential(gtfs_df, census_df)
    
    # Save sample data
    data_dir = save_sample_data(gtfs_df, census_df)
    
    print("\n" + "=" * 60)
    print("🎉 STEP 0 COMPLETE!")
    print("📁 Sample data saved in 'data/' directory")
    print("📋 Ready to proceed to STEP 1: Website Setup & File Upload")
    print("\nNext steps:")
    print("1. Review the data schemas above")
    print("2. Verify the sample data meets your requirements")
    print("3. If needed, download actual data from the provided sources")
    print("4. Give permission to proceed to STEP 1")

if __name__ == "__main__":
    main()
