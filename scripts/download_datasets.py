#!/usr/bin/env python3
"""
Dataset Download Script for TrafficFlowOpt
Downloads only approved, verified traffic datasets
"""

import os
import requests
import pandas as pd
from pathlib import Path

def create_data_dirs():
    """Create data directories if they don't exist"""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def download_file(url, filename):
    """Download a file from URL"""
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(f"data/raw/{filename}", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def download_datasets():
    """Download approved traffic datasets"""
    create_data_dirs()
    
    # Dataset URLs (Note: These are placeholder URLs - replace with actual dataset URLs)
    datasets = {
        # NYC Open Data - Automated Traffic Volume Counts
        "nyc_traffic_counts.csv": "https://data.cityofnewyork.us/api/views/7ym2-wayt/rows.csv?accessType=DOWNLOAD",
        
        # Note: For other datasets, you'll need to:
        # 1. Create accounts on respective platforms (Kaggle, etc.)
        # 2. Use their APIs or download manually
        # 3. Place files in data/raw/ directory
    }
    
    # Create dataset info file
    dataset_info = {
        "datasets": [
            {
                "name": "NYC Automated Traffic Volume Counts",
                "source": "NYC Open Data",
                "format": "CSV",
                "description": "Real traffic volumes over time and road segments",
                "file": "nyc_traffic_counts.csv",
                "url": "https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt"
            },
            {
                "name": "Urban Traffic Flow Dataset",
                "source": "Kaggle",
                "format": "CSV", 
                "description": "Temporal and spatial traffic features",
                "file": "urban_traffic_flow.csv",
                "url": "https://www.kaggle.com/datasets/hasibullahaman/urban-traffic-flow-dataset",
                "note": "Requires Kaggle account and API setup"
            },
            {
                "name": "Glasgow City Long-Term Traffic Flow",
                "source": "Glasgow Open Data",
                "format": "CSV",
                "description": "15-minute interval sensor data (2019-2023)",
                "file": "glasgow_traffic_flow.csv",
                "url": "https://open-data.glasgow.gov.uk/",
                "note": "Search for traffic flow datasets"
            },
            {
                "name": "California PeMS Traffic Speed Data",
                "source": "California Department of Transportation",
                "format": "CSV",
                "description": "State-wide vehicle speed, volume, and occupancy",
                "file": "california_pems.csv",
                "url": "http://pems.dot.ca.gov/",
                "note": "Requires registration"
            },
            {
                "name": "Bangladeshi Urban Traffic Dataset",
                "source": "Mendeley Data",
                "format": "Images/CSV",
                "description": "Annotated vehicle and pedestrian data",
                "file": "bangladeshi_traffic.zip",
                "url": "https://data.mendeley.com/datasets/",
                "note": "Search for Bangladesh traffic dataset"
            }
        ]
    }
    
    # Save dataset information
    import json
    with open("data/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Download available datasets
    for filename, url in datasets.items():
        download_file(url, filename)
    
    print("\n📊 Dataset Download Summary:")
    print("=" * 50)
    for dataset in dataset_info["datasets"]:
        print(f"• {dataset['name']}")
        print(f"  Source: {dataset['source']}")
        print(f"  Format: {dataset['format']}")
        if 'note' in dataset:
            print(f"  Note: {dataset['note']}")
        print()

if __name__ == "__main__":
    download_datasets()