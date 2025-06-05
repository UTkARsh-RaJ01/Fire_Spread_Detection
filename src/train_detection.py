"""
Training Script for Fire Detection Model
Trains a CNN model to detect fires in satellite images
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from detection_model import FireDetectionCNN, FireDetectionTrainer, evaluate_model
from preprocessing import prepare_detection_data
import torch

def main():
    print("🔥 Fire Detection Model Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'data_path': 'data/processed/fire_detection_data.csv',
        'batch_size': 16,  # Smaller batch size for beginners
        'num_epochs': 25,  # Reduced epochs for faster training
        'test_size': 0.2,
        'val_size': 0.1,
        'save_dir': 'models/saved_models'
    }
    
    try:
        # Prepare data
        print("📊 Preparing data...")
        train_loader, val_loader, test_loader = prepare_detection_data(
            data_path=config['data_path'],
            test_size=config['test_size'],
            val_size=config['val_size'],
            batch_size=config['batch_size']
        )
        
        print(f"✅ Data prepared:")
        print(f"   - Training batches: {len(train_loader)}")
        print(f"   - Validation batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
        
        # Create model
        print("\n🏗️ Creating model...")
        model = FireDetectionCNN(num_classes=1, pretrained=True)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        trainer = FireDetectionTrainer(model)
        
        # Train model
        print("\n🚀 Starting training...")
        best_model_path = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            save_dir=config['save_dir']
        )
        
        # Plot training history
        print("\n📊 Plotting training history...")
        trainer.plot_training_history(save_path="models/detection_training_history.png")
        
        # Evaluate on test set
        print("\n🧪 Evaluating on test set...")
        from detection_model import load_trained_model
        best_model = load_trained_model(best_model_path)
        test_results = evaluate_model(best_model, test_loader)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Best model saved at: {best_model_path}")
        print(f"🎯 Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"🎯 Test F1-Score: {test_results['f1_score']:.4f}")
        
        # Save training summary
        summary = {
            'model_type': 'Fire Detection CNN',
            'architecture': 'ResNet18-based',
            'epochs_trained': config['num_epochs'],
            'best_model_path': str(best_model_path),
            'test_accuracy': test_results['accuracy'],
            'test_f1_score': test_results['f1_score'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall']
        }
        
        import json
        with open('models/detection_model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n✅ Model summary saved to: models/detection_model_summary.json")
        
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        print("💡 Please run 'python src/data_collection.py' first to generate data")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 