"""
Data Processing Module for TrafficFlowOpt
Handles loading, cleaning, and processing of approved traffic datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import requests
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficDataProcessor:
    """Processes and validates approved traffic datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Load dataset information
        dataset_info_path = self.data_dir / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path) as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {"datasets": []}
    
    def download_nyc_traffic_data(self) -> bool:
        """Download NYC traffic volume counts data"""
        try:
            url = "https://data.cityofnewyork.us/api/views/7ym2-wayt/rows.csv?accessType=DOWNLOAD"
            logger.info("Downloading NYC traffic data...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            output_path = self.raw_dir / "nyc_traffic_counts.csv"
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✓ Downloaded NYC data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download NYC data: {e}")
            return False
    
    def load_nyc_traffic_data(self) -> pd.DataFrame:
        """Load and clean NYC traffic volume data"""
        file_path = self.raw_dir / "nyc_traffic_counts.csv"
        
        if not file_path.exists():
            logger.warning("NYC data not found, attempting download...")
            if not self.download_nyc_traffic_data():
                raise FileNotFoundError("Could not load or download NYC traffic data")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded NYC data: {len(df)} records")
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Parse dates if date column exists
            date_columns = [col for col in df.columns if 'date' in col or 'time' in col]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Could not parse date column: {col}")
            
            # Extract traffic flow metrics
            if 'vol' in df.columns:
                df['flow_rate'] = pd.to_numeric(df['vol'], errors='coerce')
            elif 'volume' in df.columns:
                df['flow_rate'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Add derived features
            df['hour'] = df[date_columns[0]].dt.hour if date_columns else None
            df['day_of_week'] = df[date_columns[0]].dt.dayofweek if date_columns else None
            
            # Remove invalid records
            df = df.dropna(subset=['flow_rate'])
            df = df[df['flow_rate'] >= 0]  # Remove negative flows
            
            logger.info(f"Cleaned NYC data: {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading NYC data: {e}")
            raise
    
    def generate_synthetic_glasgow_data(self) -> pd.DataFrame:
        """Generate synthetic Glasgow-style traffic data for demonstration"""
        logger.info("Generating synthetic Glasgow traffic data...")
        
        # Create 15-minute interval data for demonstration
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Generate time series with 15-minute intervals
        time_range = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        data = []
        sensor_ids = [f"SENSOR_{i:03d}" for i in range(1, 21)]  # 20 sensors
        
        for sensor_id in sensor_ids:
            base_flow = np.random.uniform(200, 800)  # Base flow rate
            
            for timestamp in time_range:
                # Add temporal patterns
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
                
                # Peak hours effect
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    flow_multiplier = 1.5
                elif 22 <= hour or hour <= 6:
                    flow_multiplier = 0.3
                else:
                    flow_multiplier = 1.0
                
                # Weekend effect
                if day_of_week >= 5:  # Weekend
                    flow_multiplier *= 0.7
                
                # Add random variation
                noise = np.random.normal(0, 0.1)
                flow_rate = base_flow * flow_multiplier * (1 + noise)
                flow_rate = max(0, flow_rate)  # Ensure non-negative
                
                # Calculate speed and occupancy from flow
                capacity = 1800  # vehicles per hour
                density = flow_rate / capacity
                
                # Speed-density relationship (Greenshields model)
                free_flow_speed = 60  # km/h
                average_speed = free_flow_speed * (1 - density)
                average_speed = max(5, average_speed)  # Minimum speed
                
                occupancy = min(0.95, density * 100)  # Percentage
                
                data.append({
                    'sensor_id': sensor_id,
                    'timestamp': timestamp,
                    'flow_rate': flow_rate,
                    'average_speed': average_speed,
                    'occupancy': occupancy,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'latitude': 55.8642 + np.random.uniform(-0.1, 0.1),
                    'longitude': -4.2518 + np.random.uniform(-0.1, 0.1)
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated synthetic Glasgow data: {len(df)} records")
        return df
    
    def generate_synthetic_california_data(self) -> pd.DataFrame:
        """Generate synthetic California PeMS-style data"""
        logger.info("Generating synthetic California PeMS data...")
        
        # Highway segments in California
        segments = [
            {'name': 'I-405_N', 'lat': 34.0522, 'lon': -118.2437},
            {'name': 'I-101_S', 'lat': 37.7749, 'lon': -122.4194},
            {'name': 'I-80_E', 'lat': 37.8044, 'lon': -122.2712},
            {'name': 'I-5_N', 'lat': 32.7157, 'lon': -117.1611},
            {'name': 'SR-99_S', 'lat': 36.7378, 'lon': -119.7871}
        ]
        
        # Generate hourly data for 1 year
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        time_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        data = []
        
        for segment in segments:
            for timestamp in time_range:
                # Multi-lane highway characteristics
                num_lanes = np.random.randint(3, 6)
                capacity_per_lane = 2000  # vehicles per hour per lane
                total_capacity = num_lanes * capacity_per_lane
                
                # Traffic patterns
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
                
                # Complex traffic patterns for highways
                if day_of_week < 5:  # Weekday
                    if 6 <= hour <= 10:  # Morning peak
                        utilization = np.random.uniform(0.7, 0.95)
                    elif 15 <= hour <= 19:  # Evening peak
                        utilization = np.random.uniform(0.8, 0.95)
                    elif 22 <= hour or hour <= 5:  # Night
                        utilization = np.random.uniform(0.1, 0.3)
                    else:
                        utilization = np.random.uniform(0.4, 0.7)
                else:  # Weekend
                    if 10 <= hour <= 16:  # Afternoon
                        utilization = np.random.uniform(0.5, 0.8)
                    else:
                        utilization = np.random.uniform(0.2, 0.5)
                
                flow_rate = total_capacity * utilization
                
                # Speed calculation using fundamental diagram
                critical_density = 45  # vehicles per km per lane
                free_flow_speed = 120  # km/h
                
                density = flow_rate / (free_flow_speed * num_lanes)
                if density < critical_density:
                    speed = free_flow_speed
                else:
                    # Congested regime
                    speed = free_flow_speed * (1 - (density - critical_density) / (180 - critical_density))
                    speed = max(20, speed)  # Minimum speed
                
                occupancy = min(95, (density / 180) * 100)  # Percentage
                
                data.append({
                    'segment_id': segment['name'],
                    'timestamp': timestamp,
                    'flow_rate': flow_rate,
                    'average_speed': speed,
                    'occupancy': occupancy,
                    'num_lanes': num_lanes,
                    'latitude': segment['lat'],
                    'longitude': segment['lon'],
                    'hour': hour,
                    'day_of_week': day_of_week
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated synthetic California data: {len(df)} records")
        return df
    
    def standardize_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize dataset to common format"""
        logger.info(f"Standardizing {dataset_name} dataset...")
        
        # Ensure required columns exist
        required_columns = ['flow_rate', 'average_speed', 'occupancy']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'average_speed' and 'speed' in df.columns:
                    df['average_speed'] = df['speed']
                elif col == 'occupancy' and 'flow_rate' in df.columns:
                    # Estimate occupancy from flow rate
                    df['occupancy'] = np.minimum(95, df['flow_rate'] / 20)
                else:
                    df[col] = 0  # Default value
        
        # Add metadata
        df['dataset_source'] = dataset_name
        df['processed_timestamp'] = datetime.now()
        
        # Data quality checks
        df = df[df['flow_rate'] >= 0]
        df = df[df['average_speed'] >= 0]
        df = df[df['occupancy'] >= 0]
        df = df[df['occupancy'] <= 100]
        
        logger.info(f"Standardized {dataset_name}: {len(df)} valid records")
        return df
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """Create unified dataset from all available sources"""
        logger.info("Creating unified traffic dataset...")
        
        datasets = []
        
        # Load NYC data if available
        try:
            nyc_df = self.load_nyc_traffic_data()
            nyc_standardized = self.standardize_dataset(nyc_df, "NYC_Traffic_Counts")
            datasets.append(nyc_standardized)
        except Exception as e:
            logger.warning(f"Could not load NYC data: {e}")
        
        # Generate synthetic data for other sources
        glasgow_df = self.generate_synthetic_glasgow_data()
        glasgow_standardized = self.standardize_dataset(glasgow_df, "Glasgow_Traffic_Flow")
        datasets.append(glasgow_standardized)
        
        california_df = self.generate_synthetic_california_data()
        california_standardized = self.standardize_dataset(california_df, "California_PeMS")
        datasets.append(california_standardized)
        
        # Combine all datasets
        if datasets:
            unified_df = pd.concat(datasets, ignore_index=True)
            
            # Sort by timestamp if available
            if 'timestamp' in unified_df.columns:
                unified_df = unified_df.sort_values('timestamp')
            
            # Save unified dataset
            output_path = self.processed_dir / "unified_traffic_data.csv"
            unified_df.to_csv(output_path, index=False)
            logger.info(f"Saved unified dataset to {output_path}")
            
            # Create summary statistics
            self.create_data_summary(unified_df)
            
            return unified_df
        else:
            raise ValueError("No datasets could be loaded")
    
    def create_data_summary(self, df: pd.DataFrame):
        """Create summary statistics and visualizations"""
        logger.info("Creating data summary...")
        
        # Summary statistics
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            },
            'flow_rate_stats': {
                'mean': float(df['flow_rate'].mean()),
                'std': float(df['flow_rate'].std()),
                'min': float(df['flow_rate'].min()),
                'max': float(df['flow_rate'].max())
            },
            'speed_stats': {
                'mean': float(df['average_speed'].mean()),
                'std': float(df['average_speed'].std()),
                'min': float(df['average_speed'].min()),
                'max': float(df['average_speed'].max())
            },
            'datasets_included': df['dataset_source'].unique().tolist()
        }
        
        # Save summary
        summary_path = self.processed_dir / "data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visualizations
        self.create_visualizations(df)
        
        logger.info(f"Data summary saved to {summary_path}")
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create data visualization plots"""
        logger.info("Creating visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flow rate distribution
        axes[0, 0].hist(df['flow_rate'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Flow Rate Distribution')
        axes[0, 0].set_xlabel('Flow Rate (vehicles/hour)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Speed distribution
        axes[0, 1].hist(df['average_speed'], bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Average Speed Distribution')
        axes[0, 1].set_xlabel('Speed (km/h)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Flow vs Speed scatter plot
        sample_df = df.sample(min(1000, len(df)))  # Sample for performance
        axes[1, 0].scatter(sample_df['flow_rate'], sample_df['average_speed'], 
                          alpha=0.5, color='coral')
        axes[1, 0].set_title('Flow Rate vs Average Speed')
        axes[1, 0].set_xlabel('Flow Rate (vehicles/hour)')
        axes[1, 0].set_ylabel('Speed (km/h)')
        
        # Dataset composition
        dataset_counts = df['dataset_source'].value_counts()
        axes[1, 1].pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Dataset Composition')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.processed_dir / "data_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {plot_path}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality and completeness"""
        logger.info("Validating data quality...")
        
        quality_report = {
            'total_records': len(df),
            'missing_data': {
                col: df[col].isnull().sum() for col in df.columns
            },
            'outliers': {},
            'data_consistency': {}
        }
        
        # Check for outliers using IQR method
        numeric_columns = ['flow_rate', 'average_speed', 'occupancy']
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                quality_report['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100
                }
        
        # Data consistency checks
        if 'flow_rate' in df.columns and 'average_speed' in df.columns:
            # Check for impossible combinations (e.g., high flow with very low speed)
            inconsistent = df[(df['flow_rate'] > df['flow_rate'].quantile(0.8)) & 
                             (df['average_speed'] < df['average_speed'].quantile(0.2))]
            quality_report['data_consistency']['flow_speed_inconsistency'] = len(inconsistent)
        
        # Save quality report
        report_path = self.processed_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info(f"Data quality report saved to {report_path}")
        return quality_report