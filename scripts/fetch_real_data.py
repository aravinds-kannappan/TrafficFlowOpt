#!/usr/bin/env python3
"""
Real Dataset Fetcher for TrafficFlowOpt
Downloads actual traffic datasets from verified public sources
"""

import os
import requests
import pandas as pd
import json
import logging
from pathlib import Path
import zipfile
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataFetcher:
    """Fetches real traffic datasets from public APIs and repositories"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_nyc_traffic_data(self):
        """Fetch real NYC traffic volume counts from NYC Open Data"""
        try:
            logger.info("Fetching NYC traffic volume counts...")
            
            # NYC Open Data API endpoint for Automated Traffic Volume Counts
            url = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
            
            # Add query parameters to limit data size and get recent data
            params = {
                "$limit": 10000,
                "$order": "date DESC",
                "$where": "date > '2023-01-01'"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched {len(data)} NYC traffic records")
            
            # Convert to DataFrame and save
            df = pd.DataFrame(data)
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Save raw data
            output_path = self.raw_dir / "nyc_traffic_counts.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Also save as CSV
            csv_path = self.raw_dir / "nyc_traffic_counts.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved NYC data to {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch NYC data: {e}")
            return None
    
    def fetch_chicago_traffic_data(self):
        """Fetch Chicago traffic data from Chicago Data Portal"""
        try:
            logger.info("Fetching Chicago traffic data...")
            
            # Chicago Data Portal - Traffic Crashes
            url = "https://data.cityofchicago.org/resource/85ca-t3if.json"
            
            params = {
                "$limit": 5000,
                "$order": "crash_date DESC",
                "$where": "crash_date > '2023-01-01'"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched {len(data)} Chicago traffic records")
            
            df = pd.DataFrame(data)
            
            # Save data
            output_path = self.raw_dir / "chicago_traffic_crashes.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            csv_path = self.raw_dir / "chicago_traffic_crashes.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved Chicago data to {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Chicago data: {e}")
            return None
    
    def fetch_seattle_traffic_data(self):
        """Fetch Seattle traffic flow data"""
        try:
            logger.info("Fetching Seattle traffic flow data...")
            
            # Seattle Traffic Flow Counts
            url = "https://data.seattle.gov/resource/tw7j-dfaw.json"
            
            params = {
                "$limit": 8000,
                "$order": "datetime DESC"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched {len(data)} Seattle traffic flow records")
            
            df = pd.DataFrame(data)
            
            # Save data
            output_path = self.raw_dir / "seattle_traffic_flow.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            csv_path = self.raw_dir / "seattle_traffic_flow.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved Seattle data to {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Seattle data: {e}")
            return None
    
    def fetch_uk_traffic_data(self):
        """Fetch UK traffic data from data.gov.uk"""
        try:
            logger.info("Fetching UK traffic count data...")
            
            # UK Department for Transport - Road Traffic Statistics
            # This is a sample URL - actual data.gov.uk APIs may require registration
            url = "https://roadtraffic.dft.gov.uk/api/downloads/major-road-traffic"
            
            # For demonstration, we'll fetch from a direct CSV link that's publicly available
            # UK traffic data from data.gov.uk
            csv_url = "https://storage.googleapis.com/road-traffic-statistics/downloads/data-gov-uk-make-data/road_traffic_by_local_authority.csv"
            
            response = requests.get(csv_url, timeout=60)
            response.raise_for_status()
            
            # Save raw CSV
            csv_path = self.raw_dir / "uk_traffic_by_authority.csv"
            with open(csv_path, 'wb') as f:
                f.write(response.content)
            
            # Load and process
            df = pd.read_csv(csv_path)
            logger.info(f"Fetched {len(df)} UK traffic records")
            
            logger.info(f"Saved UK traffic data to {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch UK data: {e}")
            return None
    
    def fetch_github_traffic_datasets(self):
        """Fetch traffic datasets from public GitHub repositories"""
        try:
            logger.info("Fetching traffic datasets from GitHub repositories...")
            
            # List of known public traffic datasets on GitHub
            datasets = [
                {
                    "name": "traffic_speed_prediction",
                    "url": "https://raw.githubusercontent.com/xiaochus/TrafficSpeedPrediction/master/data/speed.csv",
                    "filename": "github_traffic_speed.csv"
                },
                {
                    "name": "beijing_taxi_data",
                    "url": "https://raw.githubusercontent.com/TolicWang/Beijing-Taxi-Data/master/data/sample_data.csv",
                    "filename": "github_beijing_taxi.csv"
                }
            ]
            
            fetched_datasets = []
            
            for dataset in datasets:
                try:
                    logger.info(f"Fetching {dataset['name']}...")
                    response = requests.get(dataset["url"], timeout=30)
                    
                    if response.status_code == 200:
                        file_path = self.raw_dir / dataset["filename"]
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                        
                        # Try to load as DataFrame to validate
                        df = pd.read_csv(file_path)
                        logger.info(f"✓ Fetched {dataset['name']}: {len(df)} records")
                        fetched_datasets.append({
                            "name": dataset["name"],
                            "path": str(file_path),
                            "records": len(df)
                        })
                    else:
                        logger.warning(f"Failed to fetch {dataset['name']}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error fetching {dataset['name']}: {e}")
                    continue
                
                # Be respectful to servers
                time.sleep(1)
            
            return fetched_datasets
            
        except Exception as e:
            logger.error(f"Failed to fetch GitHub datasets: {e}")
            return []
    
    def fetch_opendata_traffic_feeds(self):
        """Fetch traffic data from various open data portals"""
        try:
            logger.info("Fetching from open data portals...")
            
            # San Francisco Traffic data
            try:
                sf_url = "https://data.sfgov.org/resource/ritf-b9ki.json"
                sf_params = {"$limit": 5000}
                
                sf_response = requests.get(sf_url, params=sf_params, timeout=30)
                if sf_response.status_code == 200:
                    sf_data = sf_response.json()
                    sf_df = pd.DataFrame(sf_data)
                    
                    sf_path = self.raw_dir / "san_francisco_traffic.csv"
                    sf_df.to_csv(sf_path, index=False)
                    logger.info(f"✓ Saved San Francisco traffic data: {len(sf_df)} records")
                
            except Exception as e:
                logger.warning(f"San Francisco data fetch failed: {e}")
            
            # Austin Traffic data
            try:
                austin_url = "https://data.austintexas.gov/resource/sh59-i6y9.json"
                austin_params = {"$limit": 5000}
                
                austin_response = requests.get(austin_url, params=austin_params, timeout=30)
                if austin_response.status_code == 200:
                    austin_data = austin_response.json()
                    austin_df = pd.DataFrame(austin_data)
                    
                    austin_path = self.raw_dir / "austin_traffic.csv"
                    austin_df.to_csv(austin_path, index=False)
                    logger.info(f"✓ Saved Austin traffic data: {len(austin_df)} records")
                
            except Exception as e:
                logger.warning(f"Austin data fetch failed: {e}")
            
        except Exception as e:
            logger.error(f"Open data fetch failed: {e}")
    
    def process_real_datasets(self):
        """Process and standardize all fetched real datasets"""
        logger.info("Processing real traffic datasets...")
        
        processed_datasets = []
        
        # Process each CSV file in raw directory
        for csv_file in self.raw_dir.glob("*.csv"):
            try:
                logger.info(f"Processing {csv_file.name}...")
                df = pd.read_csv(csv_file)
                
                # Standardize dataset
                processed_df = self.standardize_traffic_data(df, csv_file.stem)
                
                if processed_df is not None and len(processed_df) > 0:
                    # Save processed data
                    output_path = self.processed_dir / f"processed_{csv_file.name}"
                    processed_df.to_csv(output_path, index=False)
                    
                    processed_datasets.append({
                        "original_file": csv_file.name,
                        "processed_file": output_path.name,
                        "records": len(processed_df),
                        "columns": list(processed_df.columns),
                        "source": self.identify_data_source(csv_file.name)
                    })
                    
                    logger.info(f"✓ Processed {csv_file.name}: {len(processed_df)} records")
                
            except Exception as e:
                logger.error(f"Failed to process {csv_file.name}: {e}")
                continue
        
        # Create unified dataset
        unified_df = self.create_unified_dataset(processed_datasets)
        
        # Save processing report
        report = {
            "processing_timestamp": datetime.now().isoformat(),
            "datasets_processed": len(processed_datasets),
            "total_records": sum(d["records"] for d in processed_datasets),
            "datasets": processed_datasets,
            "unified_dataset_records": len(unified_df) if unified_df is not None else 0
        }
        
        report_path = self.processed_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing complete. Report saved to {report_path}")
        return report
    
    def standardize_traffic_data(self, df, source_name):
        """Standardize traffic data to common format"""
        try:
            # Create standardized columns
            standardized_df = pd.DataFrame()
            
            # Try to map common column patterns
            column_mapping = self.get_column_mapping(df.columns, source_name)
            
            for std_col, source_cols in column_mapping.items():
                for source_col in source_cols:
                    if source_col in df.columns:
                        if std_col == 'timestamp':
                            standardized_df[std_col] = pd.to_datetime(df[source_col], errors='coerce')
                        elif std_col in ['flow_rate', 'speed', 'volume', 'count']:
                            standardized_df[std_col] = pd.to_numeric(df[source_col], errors='coerce')
                        else:
                            standardized_df[std_col] = df[source_col]
                        break
            
            # Add metadata
            standardized_df['data_source'] = source_name
            standardized_df['processing_timestamp'] = datetime.now()
            
            # Remove rows with all NaN values
            standardized_df = standardized_df.dropna(how='all')
            
            # Remove duplicate rows
            standardized_df = standardized_df.drop_duplicates()
            
            return standardized_df
            
        except Exception as e:
            logger.error(f"Standardization failed for {source_name}: {e}")
            return None
    
    def get_column_mapping(self, columns, source_name):
        """Get column mapping based on source and available columns"""
        columns_lower = [col.lower() for col in columns]
        
        mapping = {
            'timestamp': ['date', 'datetime', 'time', 'crash_date', 'date_time'],
            'flow_rate': ['volume', 'count', 'flow', 'total_volume', 'avg_daily_traffic'],
            'speed': ['speed', 'avg_speed', 'average_speed'],
            'location': ['location', 'street', 'road', 'segment', 'intersection'],
            'latitude': ['lat', 'latitude', 'y_coordinate'],
            'longitude': ['lon', 'lng', 'longitude', 'x_coordinate'],
            'direction': ['direction', 'dir', 'bearing'],
            'vehicle_type': ['vehicle_type', 'vehicle_class', 'type']
        }
        
        # Find actual column matches
        final_mapping = {}
        for std_col, possible_cols in mapping.items():
            final_mapping[std_col] = []
            for possible_col in possible_cols:
                # Find exact or partial matches
                matches = [col for col in columns if possible_col in col.lower()]
                if matches:
                    final_mapping[std_col].extend(matches)
        
        return final_mapping
    
    def identify_data_source(self, filename):
        """Identify the source of the dataset based on filename"""
        if 'nyc' in filename.lower():
            return 'NYC Open Data'
        elif 'chicago' in filename.lower():
            return 'Chicago Data Portal'
        elif 'seattle' in filename.lower():
            return 'Seattle Open Data'
        elif 'uk' in filename.lower():
            return 'UK Government Data'
        elif 'github' in filename.lower():
            return 'GitHub Repository'
        elif 'san_francisco' in filename.lower():
            return 'San Francisco Open Data'
        elif 'austin' in filename.lower():
            return 'Austin Open Data'
        else:
            return 'Unknown Source'
    
    def create_unified_dataset(self, processed_datasets):
        """Create a unified dataset from all processed data"""
        try:
            unified_dfs = []
            
            for dataset_info in processed_datasets:
                file_path = self.processed_dir / dataset_info["processed_file"]
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    unified_dfs.append(df)
            
            if unified_dfs:
                unified_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
                
                # Save unified dataset
                unified_path = self.processed_dir / "unified_real_traffic_data.csv"
                unified_df.to_csv(unified_path, index=False)
                
                logger.info(f"Created unified dataset with {len(unified_df)} records")
                return unified_df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create unified dataset: {e}")
            return None
    
    def run_complete_fetch(self):
        """Run complete data fetching pipeline"""
        logger.info("Starting complete real data fetch pipeline...")
        
        # Fetch from all sources
        datasets_fetched = []
        
        # NYC Data
        nyc_df = self.fetch_nyc_traffic_data()
        if nyc_df is not None:
            datasets_fetched.append("NYC")
        
        # Chicago Data
        chicago_df = self.fetch_chicago_traffic_data()
        if chicago_df is not None:
            datasets_fetched.append("Chicago")
        
        # Seattle Data
        seattle_df = self.fetch_seattle_traffic_data()
        if seattle_df is not None:
            datasets_fetched.append("Seattle")
        
        # UK Data
        uk_df = self.fetch_uk_traffic_data()
        if uk_df is not None:
            datasets_fetched.append("UK")
        
        # GitHub datasets
        github_datasets = self.fetch_github_traffic_datasets()
        if github_datasets:
            datasets_fetched.extend([d["name"] for d in github_datasets])
        
        # Open data portals
        self.fetch_opendata_traffic_feeds()
        
        # Process all fetched data
        processing_report = self.process_real_datasets()
        
        logger.info("="*50)
        logger.info("DATA FETCH COMPLETE")
        logger.info(f"Sources fetched: {', '.join(datasets_fetched)}")
        logger.info(f"Total datasets: {processing_report['datasets_processed']}")
        logger.info(f"Total records: {processing_report['total_records']}")
        logger.info("="*50)
        
        return processing_report

def main():
    """Main execution function"""
    print("🚦 TrafficFlowOpt - Real Data Fetcher")
    print("=" * 50)
    print("Fetching real traffic datasets from verified sources...")
    print()
    
    fetcher = RealDataFetcher()
    
    try:
        report = fetcher.run_complete_fetch()
        
        print("\n📊 Data Fetch Summary:")
        print("-" * 30)
        print(f"✓ Datasets processed: {report['datasets_processed']}")
        print(f"✓ Total records: {report['total_records']:,}")
        print(f"✓ Data saved to: data/processed/")
        
        if report['datasets']:
            print(f"\n📋 Datasets fetched:")
            for dataset in report['datasets']:
                print(f"  • {dataset['source']}: {dataset['records']:,} records")
        
        print(f"\n🎯 Ready for TrafficFlowOpt processing!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())