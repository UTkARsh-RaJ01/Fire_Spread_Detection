"""
Utility functions for Forest Fire Detection and Spread Prediction System
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models/saved_models',
        'models/checkpoints',
        'logs',
        'outputs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_system_requirements():
    """Check if system meets requirements"""
    print("🔍 Checking system requirements...")
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check other dependencies
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'plotly', 
        'folium', 'scikit-learn', 'PIL', 'cv2'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not installed")
            return False
    
    print("🎉 All requirements satisfied!")
    return True

def generate_fire_report(detection_results=None, prediction_results=None, save_path="outputs/fire_report.json"):
    """Generate comprehensive fire monitoring report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'version': '1.0',
            'location': 'India',
            'monitoring_areas': ['Odisha', 'Chhattisgarh', 'Maharashtra', 'Telangana', 'Jharkhand']
        }
    }
    
    if detection_results:
        report['fire_detection'] = {
            'fires_detected': len([r for r in detection_results if r.get('has_fire', False)]),
            'total_images_analyzed': len(detection_results),
            'detection_rate': len([r for r in detection_results if r.get('has_fire', False)]) / len(detection_results),
            'average_confidence': np.mean([r.get('confidence', 0) for r in detection_results]),
            'detections': detection_results
        }
    
    if prediction_results:
        report['spread_prediction'] = {
            'high_risk_areas': len([r for r in prediction_results if r.get('risk_level') in ['High', 'Critical']]),
            'average_risk_score': np.mean([r.get('fire_risk_probability', 0) for r in prediction_results]),
            'predictions': prediction_results
        }
    
    # Save report
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Fire report saved to: {save_path}")
    return report

def plot_model_comparison(detection_metrics, prediction_metrics, save_path="outputs/model_comparison.png"):
    """Plot comparison of model performances"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Detection model metrics
    det_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    det_values = [
        detection_metrics.get('accuracy', 0),
        detection_metrics.get('precision', 0),
        detection_metrics.get('recall', 0),
        detection_metrics.get('f1_score', 0)
    ]
    
    bars1 = ax1.bar(det_metrics, det_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Fire Detection Model Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Prediction model metrics
    pred_metrics = ['MAE', 'RMSE', 'R² Score']
    pred_values = [
        prediction_metrics.get('mae', 0),
        prediction_metrics.get('rmse', 0),
        prediction_metrics.get('r2_score', 0)
    ]
    
    # Normalize MAE and RMSE for better visualization (assuming they're small values)
    display_values = [
        pred_values[0] * 10,  # Scale MAE
        pred_values[1] * 10,  # Scale RMSE
        pred_values[2]        # R² as is
    ]
    
    bars2 = ax2.bar(pred_metrics, display_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Fire Spread Prediction Model Performance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        original_value = pred_values[i]
        ax2.annotate(f'{original_value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Model comparison plot saved to: {save_path}")
    plt.show()

def calculate_fire_risk_score(temperature, humidity, wind_speed, rainfall, vegetation_index=0.5):
    """Calculate fire risk score based on weather conditions"""
    # Normalize inputs
    temp_norm = min(max(temperature - 15, 0), 35) / 35  # Normalize to 0-1
    humidity_norm = (100 - humidity) / 100  # Inverse humidity (dry = high risk)
    wind_norm = min(wind_speed, 50) / 50  # Normalize wind speed
    rain_norm = max(0, 1 - rainfall / 10)  # Inverse rainfall (no rain = high risk)
    
    # Weighted combination
    risk_score = (
        temp_norm * 0.3 +          # Temperature weight
        humidity_norm * 0.25 +      # Humidity weight
        wind_norm * 0.2 +          # Wind speed weight
        rain_norm * 0.15 +         # Rainfall weight
        vegetation_index * 0.1     # Vegetation dryness weight
    )
    
    return min(max(risk_score, 0), 1)  # Ensure 0-1 range

def create_weather_time_series(state, start_date, end_date, save_path=None):
    """Create realistic weather time series for a given state"""
    dates = pd.date_range(start_date, end_date, freq='D')
    weather_data = []
    
    # State-specific weather patterns (simplified)
    state_patterns = {
        'Odisha': {'base_temp': 28, 'temp_var': 6, 'humid_base': 70, 'humid_var': 20},
        'Chhattisgarh': {'base_temp': 26, 'temp_var': 8, 'humid_base': 65, 'humid_var': 25},
        'Maharashtra': {'base_temp': 25, 'temp_var': 7, 'humid_base': 60, 'humid_var': 20},
        'Telangana': {'base_temp': 27, 'temp_var': 6, 'humid_base': 65, 'humid_var': 20},
        'Jharkhand': {'base_temp': 24, 'temp_var': 8, 'humid_base': 70, 'humid_var': 25}
    }
    
    pattern = state_patterns.get(state, state_patterns['Maharashtra'])
    
    for date in dates:
        month = date.month
        
        # Seasonal adjustments
        if month in [3, 4, 5]:  # Summer
            temp_adj = 8
            humid_adj = -20
            rain_prob = 0.1
        elif month in [6, 7, 8, 9]:  # Monsoon
            temp_adj = 0
            humid_adj = 15
            rain_prob = 0.6
        else:  # Winter
            temp_adj = -5
            humid_adj = 0
            rain_prob = 0.2
        
        # Generate weather values
        temperature = max(15, min(50, np.random.normal(
            pattern['base_temp'] + temp_adj, pattern['temp_var']
        )))
        
        humidity = max(10, min(100, np.random.normal(
            pattern['humid_base'] + humid_adj, pattern['humid_var']
        )))
        
        wind_speed = max(0, np.random.exponential(12))
        rainfall = np.random.exponential(5) if np.random.random() < rain_prob else 0
        
        # Calculate fire risk
        fire_risk = calculate_fire_risk_score(temperature, humidity, wind_speed, rainfall)
        
        weather_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'state': state,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind_speed, 1),
            'rainfall': round(rainfall, 2),
            'fire_risk': round(fire_risk, 4)
        })
    
    df = pd.DataFrame(weather_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"🌤️ Weather data for {state} saved to: {save_path}")
    
    return df

def validate_model_inputs(image_tensor=None, weather_sequence=None):
    """Validate inputs for model prediction"""
    errors = []
    
    if image_tensor is not None:
        if not isinstance(image_tensor, torch.Tensor):
            errors.append("Image input must be a PyTorch tensor")
        elif len(image_tensor.shape) not in [3, 4]:
            errors.append("Image tensor must have 3 or 4 dimensions (C,H,W) or (B,C,H,W)")
        elif image_tensor.shape[-1] != 224 or image_tensor.shape[-2] != 224:
            errors.append("Image must be 224x224 pixels")
    
    if weather_sequence is not None:
        if isinstance(weather_sequence, np.ndarray):
            if len(weather_sequence.shape) != 2:
                errors.append("Weather sequence must be 2D array (sequence_length, features)")
            elif weather_sequence.shape[1] != 4:
                errors.append("Weather sequence must have 4 features (temp, humidity, wind, rainfall)")
        elif isinstance(weather_sequence, torch.Tensor):
            if len(weather_sequence.shape) not in [2, 3]:
                errors.append("Weather tensor must have 2 or 3 dimensions")
    
    if errors:
        raise ValueError("Input validation failed: " + "; ".join(errors))
    
    return True

def get_model_summary(model):
    """Get summary statistics of a PyTorch model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
        'architecture': str(model.__class__.__name__)
    }
    
    return summary

def log_prediction(prediction_type, input_data, result, log_file="logs/predictions.log"):
    """Log prediction results for monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': prediction_type,
        'input_shape': str(input_data.shape) if hasattr(input_data, 'shape') else 'N/A',
        'result': result
    }
    
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def calculate_system_health():
    """Calculate overall system health score"""
    health_score = 100
    issues = []
    
    # Check if models exist
    detection_model_path = Path("models/saved_models/best_fire_detection_model.pth")
    prediction_model_path = Path("models/saved_models/best_fire_prediction_model.pth")
    
    if not detection_model_path.exists():
        health_score -= 30
        issues.append("Detection model not found")
    
    if not prediction_model_path.exists():
        health_score -= 30
        issues.append("Prediction model not found")
    
    # Check if data exists
    fire_data_path = Path("data/processed/fire_detection_data.csv")
    weather_data_path = Path("data/processed/weather_data.csv")
    
    if not fire_data_path.exists():
        health_score -= 20
        issues.append("Fire detection data not found")
    
    if not weather_data_path.exists():
        health_score -= 20
        issues.append("Weather data not found")
    
    return {
        'health_score': max(0, health_score),
        'status': 'Healthy' if health_score >= 80 else 'Warning' if health_score >= 50 else 'Critical',
        'issues': issues
    }

if __name__ == "__main__":
    print("🛠️ Forest Fire System Utilities")
    print("=" * 40)
    
    # Setup directories
    setup_directories()
    
    # Check requirements
    check_system_requirements()
    
    # Calculate system health
    health = calculate_system_health()
    print(f"\n🏥 System Health: {health['health_score']}/100 ({health['status']})")
    if health['issues']:
        print("Issues found:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    print("\n✅ Utilities check completed!") 