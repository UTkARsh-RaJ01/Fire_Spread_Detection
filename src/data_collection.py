"""
Forest Fire Data Collection Script
Downloads and prepares datasets for fire detection and spread prediction in India
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import json
from tqdm import tqdm

class ForestFireDataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_name, download_path):
        """Download dataset from Kaggle using API"""
        try:
            import kaggle
            print(f"Downloading {dataset_name}...")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=download_path, 
                unzip=True
            )
            print(f"✅ Downloaded {dataset_name}")
            return True
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def create_sample_fire_data(self):
        """Create sample fire detection data for India if Kaggle data unavailable"""
        print("Creating sample fire detection data...")
        
        # Sample satellite image data (simulated)
        sample_data = []
        india_coords = {
            'lat_min': 8.0, 'lat_max': 37.0,
            'lon_min': 68.0, 'lon_max': 97.0
        }
        
        for i in range(1000):
            lat = np.random.uniform(india_coords['lat_min'], india_coords['lat_max'])
            lon = np.random.uniform(india_coords['lon_min'], india_coords['lon_max'])
            
            # Simulate fire probability based on region
            fire_prob = 0.1  # Base probability
            if 20 < lat < 30:  # Higher fire regions
                fire_prob = 0.3
            
            has_fire = np.random.random() < fire_prob
            
            sample_data.append({
                'image_id': f'IMG_{i:04d}',
                'latitude': lat,
                'longitude': lon,
                'has_fire': int(has_fire),
                'fire_intensity': np.random.uniform(0, 1) if has_fire else 0,
                'date': pd.date_range('2020-01-01', periods=1000)[i].strftime('%Y-%m-%d')
            })
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.processed_dir / 'fire_detection_data.csv', index=False)
        print(f"✅ Created sample fire data: {len(df)} records")
        return df
    
    def create_sample_weather_data(self):
        """Create sample weather data for spread prediction"""
        print("Creating sample weather data...")
        
        # Generate weather data for major fire-prone states in India
        states = ['Odisha', 'Chhattisgarh', 'Maharashtra', 'Telangana', 'Jharkhand']
        weather_data = []
        
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        for state in states:
            for date in dates:
                month = date.month
                
                # Seasonal temperature patterns
                if month in [3, 4, 5]:  # Summer
                    temp = np.random.normal(35, 5)
                    humidity = np.random.normal(40, 10)
                elif month in [6, 7, 8, 9]:  # Monsoon
                    temp = np.random.normal(28, 3)
                    humidity = np.random.normal(80, 10)
                else:  # Winter
                    temp = np.random.normal(25, 5)
                    humidity = np.random.normal(60, 15)
                
                weather_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'state': state,
                    'temperature': max(15, min(50, temp)),
                    'humidity': max(10, min(100, humidity)),
                    'wind_speed': np.random.uniform(5, 25),
                    'rainfall': np.random.exponential(2) if month in [6, 7, 8, 9] else np.random.exponential(0.5),
                    'fire_risk': np.random.uniform(0, 1)
                })
        
        df = pd.DataFrame(weather_data)
        df.to_csv(self.processed_dir / 'weather_data.csv', index=False)
        print(f"✅ Created weather data: {len(df)} records")
        return df
    
    def download_fire_datasets(self):
        """Download popular fire datasets from Kaggle"""
        datasets = [
            'phylake1337/fire-dataset',
            'elmadafri/the-wildfire-dataset',
            'rtatman/188-million-us-wildfires'
        ]
        
        downloaded_any = False
        for dataset in datasets:
            success = self.download_kaggle_dataset(dataset, self.raw_dir)
            if success:
                downloaded_any = True
        
        if not downloaded_any:
            print("⚠️  Kaggle datasets not available, creating sample data...")
            self.create_sample_fire_data()
            self.create_sample_weather_data()
    
    def create_india_fire_coordinates(self):
        """Create coordinates for major fire-prone areas in India"""
        fire_coordinates = {
            'Odisha': {'lat': 20.9517, 'lon': 85.0985, 'fire_frequency': 'High'},
            'Chhattisgarh': {'lat': 21.2787, 'lon': 81.8661, 'fire_frequency': 'High'},
            'Maharashtra': {'lat': 19.7515, 'lon': 75.7139, 'fire_frequency': 'Medium'},
            'Telangana': {'lat': 18.1124, 'lon': 79.0193, 'fire_frequency': 'Medium'},
            'Jharkhand': {'lat': 23.6102, 'lon': 85.2799, 'fire_frequency': 'High'},
            'West Bengal': {'lat': 22.9868, 'lon': 87.8550, 'fire_frequency': 'Medium'},
            'Andhra Pradesh': {'lat': 15.9129, 'lon': 79.7400, 'fire_frequency': 'Low'},
            'Assam': {'lat': 26.2006, 'lon': 92.9376, 'fire_frequency': 'Medium'},
        }
        
        with open(self.processed_dir / 'india_fire_coordinates.json', 'w') as f:
            json.dump(fire_coordinates, f, indent=2)
        
        print(f"✅ Created India fire coordinates data")
        return fire_coordinates
    
    def collect_all_data(self):
        """Main method to collect all required data"""
        print("🔥 Starting Forest Fire Data Collection for India...")
        print("=" * 50)
        
        # Download fire datasets
        self.download_fire_datasets()
        
        # Create India-specific coordinate data
        self.create_india_fire_coordinates()
        
        # Generate sample data if needed
        if not (self.processed_dir / 'fire_detection_data.csv').exists():
            self.create_sample_fire_data()
        
        if not (self.processed_dir / 'weather_data.csv').exists():
            self.create_sample_weather_data()
        
        print("\n✅ Data collection completed!")
        print(f"📁 Data saved in: {self.data_dir}")
        print("\nNext steps:")
        print("1. Run: python src/train_detection.py")
        print("2. Run: python src/train_prediction.py")
        print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    collector = ForestFireDataCollector()
    collector.collect_all_data() 