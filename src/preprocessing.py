"""
Data Preprocessing Utilities for Forest Fire Detection and Spread Prediction
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class FireImageDataset(Dataset):
    """Custom dataset for fire detection from satellite images"""
    
    def __init__(self, data_df, transform=None, image_size=224):
        self.data = data_df
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Generate synthetic satellite image based on fire presence
        image = self.generate_synthetic_image(
            has_fire=row['has_fire'],
            fire_intensity=row.get('fire_intensity', 0),
            lat=row['latitude'],
            lon=row['longitude']
        )
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['has_fire'], dtype=torch.float32)
        
        return image, label
    
    def generate_synthetic_image(self, has_fire, fire_intensity, lat, lon):
        """Generate synthetic satellite image for training"""
        # Create base landscape image
        image = np.random.randint(50, 150, (self.image_size, self.image_size, 3), dtype=np.int32)
        
        # Add vegetation patterns (green channels)
        vegetation = np.random.randint(0, 50, (self.image_size, self.image_size))
        image[:, :, 1] = image[:, :, 1] + vegetation
        
        if has_fire:
            # Add fire signature (red/orange channels)
            fire_region_size = int(fire_intensity * 50 + 10)
            center_x, center_y = self.image_size // 2, self.image_size // 2
            
            # Create fire region
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= fire_region_size**2
            
            image[mask, 0] = np.minimum(255, image[mask, 0] + 100)  # Red
            image[mask, 1] = np.maximum(0, image[mask, 1] - 50)     # Less green
            image[mask, 2] = np.minimum(255, image[mask, 2] + 50)   # Some blue (smoke)
        
        # Convert to uint8 and ensure valid range
        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

class WeatherDataPreprocessor:
    """Preprocessor for weather data used in spread prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = ['temperature', 'humidity', 'wind_speed', 'rainfall']
        
    def prepare_time_series_data(self, df, sequence_length=7):
        """Prepare time series data for LSTM model"""
        # Sort by date and state
        df = df.sort_values(['state', 'date']).reset_index(drop=True)
        
        # Encode categorical variables
        if 'state' in df.columns:
            if 'state' not in self.label_encoders:
                self.label_encoders['state'] = LabelEncoder()
                df['state_encoded'] = self.label_encoders['state'].fit_transform(df['state'])
            else:
                df['state_encoded'] = self.label_encoders['state'].transform(df['state'])
        
        # Scale numerical features
        for feature in self.feature_columns:
            if feature in df.columns:
                if feature not in self.scalers:
                    self.scalers[feature] = StandardScaler()
                    df[f'{feature}_scaled'] = self.scalers[feature].fit_transform(df[[feature]])
                else:
                    df[f'{feature}_scaled'] = self.scalers[feature].transform(df[[feature]])
        
        # Create sequences
        sequences = []
        targets = []
        
        for state in df['state'].unique():
            state_data = df[df['state'] == state].reset_index(drop=True)
            
            for i in range(len(state_data) - sequence_length):
                sequence_features = []
                for feature in self.feature_columns:
                    if f'{feature}_scaled' in state_data.columns:
                        sequence_features.append(state_data[f'{feature}_scaled'].iloc[i:i+sequence_length].values)
                
                if sequence_features:
                    sequence = np.column_stack(sequence_features)
                    sequences.append(sequence)
                    targets.append(state_data['fire_risk'].iloc[i+sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def save_preprocessors(self, save_dir):
        """Save scalers and encoders for later use"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, save_dir / f'{name}_scaler.pkl')
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, save_dir / f'{name}_encoder.pkl')
    
    def load_preprocessors(self, save_dir):
        """Load saved scalers and encoders"""
        save_dir = Path(save_dir)
        
        # Load scalers
        for feature in self.feature_columns:
            scaler_path = save_dir / f'{feature}_scaler.pkl'
            if scaler_path.exists():
                self.scalers[feature] = joblib.load(scaler_path)
        
        # Load label encoders
        encoder_path = save_dir / 'state_encoder.pkl'
        if encoder_path.exists():
            self.label_encoders['state'] = joblib.load(encoder_path)

class DataAugmentation:
    """Image augmentation for fire detection training"""
    
    @staticmethod
    def get_train_transforms(image_size=224):
        """Get augmentation transforms for training"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms(image_size=224):
        """Get transforms for validation/testing"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def prepare_detection_data(data_path, test_size=0.2, val_size=0.1, batch_size=32):
    """Prepare data loaders for fire detection model"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=test_size+val_size, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size+val_size), random_state=42)
    
    # Create datasets
    train_dataset = FireImageDataset(
        train_df, 
        transform=DataAugmentation.get_train_transforms()
    )
    val_dataset = FireImageDataset(
        val_df, 
        transform=DataAugmentation.get_val_transforms()
    )
    test_dataset = FireImageDataset(
        test_df, 
        transform=DataAugmentation.get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def prepare_prediction_data(data_path, sequence_length=7, test_size=0.2):
    """Prepare data for spread prediction model"""
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Preprocess
    preprocessor = WeatherDataPreprocessor()
    X, y = preprocessor.prepare_time_series_data(df, sequence_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    print("🔄 Testing preprocessing pipeline...")
    
    # Test detection data preparation
    try:
        train_loader, val_loader, test_loader = prepare_detection_data("data/processed/fire_detection_data.csv")
        print(f"✅ Detection data prepared: {len(train_loader)} train batches")
    except Exception as e:
        print(f"❌ Detection data preparation failed: {e}")
    
    # Test prediction data preparation
    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_prediction_data("data/processed/weather_data.csv")
        print(f"✅ Prediction data prepared: {X_train.shape} train sequences")
    except Exception as e:
        print(f"❌ Prediction data preparation failed: {e}") 