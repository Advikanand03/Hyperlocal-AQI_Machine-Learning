"""
Hyperlocal AQI Prediction System - Data Pipeline
================================================
This script builds a comprehensive dataset by:
1. Loading grid centroids and processed station data
2. Computing dew point from temperature and humidity
3. Interpolating weather features to grid centroids using IDW
4. Mapping stations to grid cells using KDTree
5. Merging interpolated weather with PM2.5 data

Output: Final dataset ready for ML training
"""

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_dew_point(temp, rh):
    """
    Compute dew point from temperature and relative humidity.
    
    Uses Magnus formula: 
    alpha = ((a * temp) / (b + temp)) + ln(rh / 100.0)
    dwpt = (b * alpha) / (a - alpha)
    
    Args:
        temp: Temperature in °C (float or array)
        rh: Relative humidity in % (float or array)
    
    Returns:
        Dew point in °C (float or array)
    """
    a = 17.27
    b = 237.7
    
    alpha = (a * temp / (b + temp)) + np.log(rh / 100.0)
    dwpt = (b * alpha) / (a - alpha)
    
    return dwpt


def idw_interpolation(points, values, grid_points, power=2, radius=None):
    """
    Inverse Distance Weighting (IDW) interpolation.
    
    Args:
        points: Station coordinates (N, 2) - [lat, lon]
        values: Values at stations (N,) - weather feature
        grid_points: Grid centroid coordinates (M, 2) - [lat, lon]
        power: IDW power (default: 2)
        radius: Maximum distance for interpolation (None = use all points)
    
    Returns:
        Interpolated values at grid points (M,)
    """
    # Handle missing values in input
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.full(len(grid_points), np.nan)
    
    valid_points = points[valid_mask]
    valid_values = values[valid_mask]
    
    # Build KDTree for stations
    tree = KDTree(valid_points)
    
    # Query distances to all stations
    distances, indices = tree.query(grid_points, k=len(valid_points))
    
    interpolated = np.zeros(len(grid_points))
    
    for i, (dists, inds) in enumerate(zip(distances, indices)):
        if radius is not None:
            # Filter by radius
            mask = dists <= radius
            dists = dists[mask]
            inds = inds[mask]
            if len(dists) == 0:
                interpolated[i] = np.nan
                continue
        
        # Handle zero distance (exact match)
        zero_mask = dists == 0
        if np.any(zero_mask):
            interpolated[i] = valid_values[inds[zero_mask]][0]
        else:
            # IDW formula
            weights = 1.0 / (dists ** power)
            weights /= weights.sum()
            interpolated[i] = np.sum(weights * valid_values[inds])
    
    return interpolated


class HyperlocalAQIDataset:
    """Build hyperlocal AQI prediction dataset using processed station data."""
    
    def __init__(self, grid_path, processed_data_path, output_path):
        """
        Initialize dataset builder.
        
        Args:
            grid_path: Path to grid centroids CSV
            processed_data_path: Path to processed station data
            output_path: Path to save final dataset
        """
        self.grid_path = grid_path
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        
        self.grid_df = None
        self.station_data = None
        self.final_df = None
        self.grid_coords = None
    
    def step1_load_grid(self):
        """STEP 1: Load grid CSV into pandas dataframe."""
        logger.info("STEP 1: Loading grid centroids...")
        self.grid_df = pd.read_csv(self.grid_path)
        self.grid_coords = self.grid_df[['centroid_lat', 'centroid_lon']].values
        logger.info(f"Loaded {len(self.grid_df)} grid cells")
        logger.info(f"Grid columns: {self.grid_df.columns.tolist()}")
        return self.grid_df
    
    def step2_load_processed_data(self):
        """STEP 2: Load processed station dataset."""
        logger.info("STEP 2: Loading processed station data...")
        self.station_data = pd.read_csv(self.processed_data_path)
        
        # Parse timestamp and align to the hourly bucket
        self.station_data['timestamp'] = pd.to_datetime(self.station_data['timestamp']).dt.floor('H')
        
        logger.info(f"Loaded {len(self.station_data)} station records")
        logger.info(f"Columns: {self.station_data.columns.tolist()}")
        logger.info(f"Date range: {self.station_data['timestamp'].min()} to {self.station_data['timestamp'].max()}")
        logger.info(f"Unique stations: {self.station_data['station'].nunique()}")
        
        return self.station_data
    
    def step4_attach_pm25_to_grid(self):
        """Attach real PM2.5 from stations to nearest grid cell."""
        
        logger.info("STEP 4: Mapping station PM2.5 to grid cells...")
        
        # Build KDTree for grid
        tree = KDTree(self.grid_coords)
        
        # Map each station record to nearest grid
        station_coords = self.station_data[['latitude', 'longitude']].values
        _, grid_indices = tree.query(station_coords)
        
        self.station_data['grid_id'] = self.grid_df.iloc[grid_indices]['grid_id'].values
        
        # Select relevant columns
        pm25_df = self.station_data[['grid_id', 'timestamp', 'pm25', 'station']].copy()

        pm25_df = pm25_df.rename(columns={
            'timestamp': 'time',
            'pm25': 'PM2.5',
            'station': 'station_name'
        })
        
        # Merge with interpolated weather
        self.final_df = self.final_df.merge(
            pm25_df,
            on=['grid_id', 'time'],
            how='inner'
        )
        
        logger.info(f"Dataset after attaching PM2.5: {self.final_df.shape}")
        
        return self.final_df



    def step3_interpolate_weather_to_grid(self):
        """
        STEP 3: Interpolate weather features to grid centroids using IDW.
        
        For each timestamp, use all available stations to interpolate:
        - Temperature (AT)
        - Relative humidity (RH)
        - Wind speed (WS)
        - Rainfall (RF)
        - PM2.5
        """
        logger.info("STEP 3: Interpolating weather features to grid centroids using IDW...")
        
        # Get unique timestamps
        timestamps = self.station_data['timestamp'].unique()
        logger.info(f"Processing {len(timestamps)} unique timestamps...")
        
        results = []
        
        for timestamp in tqdm(timestamps, desc="Interpolating"):
            timestamp = pd.to_datetime(timestamp).floor('H')
            # Get all station data for this timestamp
            ts_data = self.station_data[self.station_data['timestamp'] == timestamp].copy()
            
            if len(ts_data) < 2:
                continue
            
            # Extract station coordinates
            station_coords = ts_data[['latitude', 'longitude']].values
            station_names = ts_data['station'].values
            station_tree = KDTree(station_coords)
            _, nearest_station_indices = station_tree.query(self.grid_coords)
            nearest_station_names = station_names[nearest_station_indices]
            
            # Interpolate each weather feature
            temp_values = ts_data['temperature'].values
            rh_values = ts_data['humidity'].values
            wspd_values = ts_data['wind_speed'].values
            prcp_values = ts_data.get('rainfall', pd.Series([0] * len(ts_data))).values if 'rainfall' in ts_data.columns else np.zeros(len(ts_data))
            # pm25_values = ts_data['pm25'].values
            # pm25_interp = idw_interpolation(station_coords, pm25_values, self.grid_coords, power=2)
            
            # IDW interpolation to grid
            temp_interp = idw_interpolation(station_coords, temp_values, self.grid_coords, power=2)
            rh_interp = idw_interpolation(station_coords, rh_values, self.grid_coords, power=2)
            wspd_interp = idw_interpolation(station_coords, wspd_values, self.grid_coords, power=2)
            prcp_interp = idw_interpolation(station_coords, prcp_values, self.grid_coords, power=2)
            # pm25_interp = idw_interpolation(station_coords, pm25_values, self.grid_coords, power=2)
            
            # Compute dew point from interpolated temp and RH
            dwpt_interp = compute_dew_point(temp_interp, rh_interp)
            
            # Create dataframe for this timestamp
            for grid_idx, grid_id in enumerate(self.grid_df['grid_id'].values):
                results.append({
                    'grid_id': grid_id,
                    'time': timestamp,
                    # 'station_name': nearest_station_names[grid_idx],
                    'temp': temp_interp[grid_idx],
                    'dwpt': dwpt_interp[grid_idx],
                    'wspd': wspd_interp[grid_idx],
                    'prcp': prcp_interp[grid_idx],
                    # 'PM2.5': np.nan
                })
        
        self.final_df = pd.DataFrame(results)
        
        logger.info(f"Final dataset shape: {self.final_df.shape}")
        logger.info(f"Columns: {self.final_df.columns.tolist()}")
        logger.info(f"Missing values:\n{self.final_df.isnull().sum()}")
        
        return self.final_df
    
    def handle_missing_values(self):
        """Handle missing values in the final dataset."""
        logger.info("Handling missing values...")
        
        # Drop rows where PM2.5 is missing (use only real labels)
        self.final_df = self.final_df.dropna(subset=['PM2.5'])

        # Fill weather features only (no PM2.5 filling)
        for col in ['temp', 'dwpt', 'wspd', 'prcp']:
            self.final_df[col] = self.final_df.groupby('grid_id')[col].transform(
                lambda x: x.fillna(method='bfill').fillna(method='ffill')
            )

        # Fill any remaining weather NaNs with mean
        for col in ['temp', 'dwpt', 'wspd', 'prcp']:
            self.final_df[col] = self.final_df[col].fillna(self.final_df[col].mean())
        
        logger.info(f"Missing values after handling:\n{self.final_df.isnull().sum()}")
        
        return self.final_df
    
    def add_time_and_lag_features(self):
        """Add time-based and lag features for ML."""
        logger.info("Adding time and lag features...")

        # Ensure time is datetime
        self.final_df['time'] = pd.to_datetime(self.final_df['time'])

        # Time features
        self.final_df['hour'] = self.final_df['time'].dt.hour
        self.final_df['day_of_week'] = self.final_df['time'].dt.dayofweek
        self.final_df['month'] = self.final_df['time'].dt.month

        # Sort for lag creation
        self.final_df = self.final_df.sort_values(['grid_id', 'time'])

        # Lag features for PM2.5 (short + medium + daily memory)
        for lag in [1, 2, 3, 6, 12, 24]:
            self.final_df[f'pm25_lag_{lag}'] = self.final_df.groupby('grid_id')['PM2.5'].shift(lag)

        # Lag features for weather (short memory)
        for col in ['temp', 'dwpt', 'wspd', 'prcp']:
            for lag in [1, 3]:
                self.final_df[f'{col}_lag_{lag}'] = self.final_df.groupby('grid_id')[col].shift(lag)

        # Rolling features (captures trends)
        self.final_df['pm25_roll_mean_3'] = self.final_df.groupby('grid_id')['PM2.5'].transform(lambda x: x.rolling(3).mean())
        self.final_df['pm25_roll_mean_6'] = self.final_df.groupby('grid_id')['PM2.5'].transform(lambda x: x.rolling(6).mean())
        self.final_df['pm25_roll_std_6'] = self.final_df.groupby('grid_id')['PM2.5'].transform(lambda x: x.rolling(6).std())

        # Drop rows with NaNs created by lagging
        self.final_df = self.final_df.dropna()

        logger.info(f"Dataset after feature engineering: {self.final_df.shape}")

        return self.final_df

    def save_dataset(self):
        """Save final dataset to CSV."""
        logger.info(f"Saving dataset to {self.output_path}...")
        self.final_df.to_csv(self.output_path, index=False)
        logger.info(f"Dataset saved successfully! Records: {len(self.final_df)}")
        logger.info("\nDataset Preview:")
        logger.info(self.final_df.head(10))
        logger.info("\nDataset Statistics:")
        logger.info(self.final_df[['temp', 'dwpt', 'wspd', 'prcp', 'PM2.5']].describe())
    
    def run_pipeline(self):
        """Run complete pipeline."""
        logger.info("=" * 80)
        logger.info("HYPERLOCAL AQI PREDICTION DATASET PIPELINE")
        logger.info("Using IDW Interpolation for Weather Features")
        logger.info("=" * 80)
        
        self.step1_load_grid()
        self.step2_load_processed_data()
        self.step3_interpolate_weather_to_grid()
        self.step4_attach_pm25_to_grid() 
        self.handle_missing_values()
        self.add_time_and_lag_features()
        self.save_dataset()
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)


def main():
    """Main execution."""
    # Define paths
    grid_path = 'outputs/bengaluru_grid_1000m_centroids.csv'
    processed_data_path = 'data/processed/bangalore_processed.csv'
    output_path = 'outputs/hyperlocal_aqi_dataset_idw.csv'
    
    # Create dataset builder
    builder = HyperlocalAQIDataset(
        grid_path=grid_path,
        processed_data_path=processed_data_path,
        output_path=output_path
    )
    
    # Run pipeline
    builder.run_pipeline()


if __name__ == '__main__':
    main()
