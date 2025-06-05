"""
Training Script for Fire Spread Prediction Model
Trains an LSTM model to predict fire spread based on weather data
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from prediction_model import FireSpreadLSTM, FireSpreadTrainer, evaluate_prediction_model
from preprocessing import prepare_prediction_data
import torch
import joblib

def main():
    print("🌪️ Fire Spread Prediction Model Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'data_path': 'data/processed/weather_data.csv',
        'sequence_length': 7,  # Use 7 days of weather data
        'num_epochs': 50,      # More epochs for LSTM
        'batch_size': 64,
        'test_size': 0.2,
        'input_size': 4,       # temperature, humidity, wind_speed, rainfall
        'hidden_size': 64,
        'num_layers': 2,
        'save_dir': 'models/saved_models'
    }
    
    try:
        # Prepare data
        print("📊 Preparing time series data...")
        X_train, X_test, y_train, y_test, preprocessor = prepare_prediction_data(
            data_path=config['data_path'],
            sequence_length=config['sequence_length'],
            test_size=config['test_size']
        )
        
        print(f"✅ Data prepared:")
        print(f"   - Training sequences: {X_train.shape}")
        print(f"   - Test sequences: {X_test.shape}")
        print(f"   - Sequence length: {config['sequence_length']} days")
        print(f"   - Features per day: {config['input_size']}")
        
        # Save preprocessor for later use
        preprocessor.save_preprocessors(config['save_dir'])
        print(f"✅ Preprocessor saved to: {config['save_dir']}")
        
        # Create model
        print("\n🏗️ Creating LSTM model...")
        model = FireSpreadLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            dropout=0.2
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Architecture: {config['num_layers']} LSTM layers, {config['hidden_size']} hidden units")
        
        # Create trainer
        trainer = FireSpreadTrainer(model)
        
        # Train model
        print("\n🚀 Starting training...")
        best_model_path = trainer.train(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            save_dir=config['save_dir']
        )
        
        # Plot training history
        print("\n📊 Plotting training history...")
        trainer.plot_training_history(save_path="models/prediction_training_history.png")
        
        # Evaluate on test set
        print("\n🧪 Final evaluation on test set...")
        from prediction_model import load_trained_prediction_model
        best_model = load_trained_prediction_model(best_model_path, input_size=config['input_size'])
        test_results = evaluate_prediction_model(best_model, X_test, y_test)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Best model saved at: {best_model_path}")
        print(f"🎯 Test MAE: {test_results['mae']:.6f}")
        print(f"🎯 Test RMSE: {test_results['rmse']:.6f}")
        print(f"🎯 Test R²: {test_results['r2_score']:.6f}")
        
        # Save training summary
        summary = {
            'model_type': 'Fire Spread Prediction LSTM',
            'architecture': f'{config["num_layers"]} LSTM layers, {config["hidden_size"]} hidden units',
            'sequence_length': config['sequence_length'],
            'input_features': config['input_size'],
            'epochs_trained': config['num_epochs'],
            'best_model_path': str(best_model_path),
            'test_mae': test_results['mae'],
            'test_rmse': test_results['rmse'],
            'test_r2_score': test_results['r2_score']
        }
        
        import json
        with open('models/prediction_model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n✅ Model summary saved to: models/prediction_model_summary.json")
        
        # Demo prediction
        print("\n🔮 Demo prediction:")
        try:
            # Create sample weather sequence for prediction
            import numpy as np
            sample_weather = np.random.rand(config['sequence_length'], config['input_size'])
            sample_weather = torch.FloatTensor(sample_weather)
            
            from prediction_model import predict_fire_spread
            prediction = predict_fire_spread(best_model, sample_weather)
            
            print(f"   Sample prediction:")
            print(f"   - Fire risk probability: {prediction['fire_risk_probability']:.4f}")
            print(f"   - Risk level: {prediction['risk_level']}")
            print(f"   - Confidence: {prediction['confidence']:.4f}")
            
        except Exception as e:
            print(f"   Demo prediction failed: {e}")
        
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        print("💡 Please run 'python src/data_collection.py' first to generate data")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 