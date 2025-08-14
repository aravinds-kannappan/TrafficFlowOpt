# Dataset Information

This directory contains real-world traffic datasets used by TrafficFlowOpt. We use only verified, publicly available datasets - no simulated or random data.

## Approved Datasets

### 1. NYC Automated Traffic Volume Counts
- **Source**: NYC Open Data
- **Format**: CSV
- **Description**: Real traffic volumes over time and road segments
- **URL**: https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt
- **Usage**: Real-time flow monitoring and baseline traffic patterns

### 2. Urban Traffic Flow Dataset (Kaggle)
- **Source**: Kaggle
- **Format**: CSV
- **Description**: Temporal and spatial traffic features
- **URL**: https://www.kaggle.com/datasets/hasibullahaman/urban-traffic-flow-dataset
- **Usage**: Machine learning model training for flow prediction
- **Note**: Requires Kaggle account

### 3. Glasgow City Long-Term Traffic Flow
- **Source**: Glasgow Open Data
- **Format**: CSV
- **Description**: 15-minute interval sensor data (2019-2023)
- **URL**: https://open-data.glasgow.gov.uk/
- **Usage**: Long-term trend analysis and seasonal pattern detection

### 4. California PeMS Traffic Speed Data
- **Source**: California Department of Transportation
- **Format**: CSV
- **Description**: State-wide vehicle speed, volume, and occupancy
- **URL**: http://pems.dot.ca.gov/
- **Usage**: Large-scale traffic optimization and validation
- **Note**: Requires registration

### 5. Bangladeshi Urban Traffic Dataset
- **Source**: Mendeley Data
- **Format**: Images/CSV
- **Description**: Annotated vehicle and pedestrian data
- **URL**: https://data.mendeley.com/datasets/
- **Usage**: Computer vision and traffic classification
- **Note**: Mixed urban traffic patterns

## Directory Structure

```
data/
├── raw/           # Original downloaded datasets
├── processed/     # Cleaned and processed data
└── dataset_info.json  # Metadata about all datasets
```

## Data Processing Pipeline

1. **Download** - Use `scripts/download_datasets.py`
2. **Validation** - Check data integrity and format
3. **Cleaning** - Remove inconsistencies and fill gaps
4. **Transformation** - Convert to standardized format
5. **Integration** - Combine datasets for analysis

## Citation Requirements

When using these datasets, please cite the original sources:
- NYC Open Data Portal
- Kaggle contributors
- Glasgow City Council
- California Department of Transportation
- Mendeley Data contributors