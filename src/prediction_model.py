"""
Fire Spread Prediction Model using LSTM
Predicts fire spread patterns based on weather and environmental data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib

class FireSpreadLSTM(nn.Module):
    """LSTM model for fire spread prediction"""
    
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(FireSpreadLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()  # Output between 0 and 1 for fire risk probability
        )
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use the last output for prediction
        output = self.fc(lstm_out[:, -1, :])
        
        return output

class FireSpreadTrainer:
    """Trainer class for fire spread prediction model"""
    
    def __init__(self, model, device='auto'):
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"🌪️ Using device: {self.device}")
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_mae_scores = []
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create batches
        dataset_size = X_train.size(0)
        indices = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices].to(self.device)
            batch_y = y_train[batch_indices].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, X_val, y_val):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            
            outputs = self.model(X_val).squeeze()
            loss = self.criterion(outputs, y_val)
            val_loss = loss.item()
            
            all_predictions = outputs.cpu().numpy()
            all_targets = y_val.cpu().numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        
        return val_loss, mae, mse, rmse, r2, all_predictions, all_targets
    
    def train(self, X_train, X_test, y_train, y_test, num_epochs=100, batch_size=32, save_dir="models/saved_models"):
        """Complete training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        best_model_path = save_dir / "best_fire_prediction_model.pth"
        
        print("🌪️ Starting Fire Spread Prediction Model Training...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, mae, mse, rmse, r2, _, _ = self.validate(X_test, y_test)
            self.val_losses.append(val_loss)
            self.val_mae_scores.append(mae)
            
            # Print metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}]")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")
                print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'mae': mae
                }, best_model_path)
                
                if (epoch + 1) % 10 == 0:
                    print(f"✅ New best model saved! Val Loss: {val_loss:.6f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Model saved at: {best_model_path}")
        
        # Save final model
        final_model_path = save_dir / "final_fire_prediction_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mae_scores': self.val_mae_scores
        }, final_model_path)
        
        return best_model_path
    
    def plot_training_history(self, save_path="models/prediction_training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Plot MAE
        ax2.plot(self.val_mae_scores, label='Validation MAE', color='green')
        ax2.set_title('Validation Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training history saved to: {save_path}")

def load_trained_prediction_model(model_path, input_size=4, device='auto'):
    """Load a trained fire spread prediction model"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FireSpreadLSTM(input_size=input_size)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Prediction model loaded from {model_path}")
    return model

def predict_fire_spread(model, weather_sequence, device='auto'):
    """Predict fire spread risk for given weather sequence"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        if isinstance(weather_sequence, np.ndarray):
            weather_sequence = torch.FloatTensor(weather_sequence)
        
        weather_sequence = weather_sequence.to(device)
        if len(weather_sequence.shape) == 2:
            weather_sequence = weather_sequence.unsqueeze(0)  # Add batch dimension
        
        prediction = model(weather_sequence)
        fire_risk = prediction.item()
        
        # Categorize risk level
        if fire_risk < 0.3:
            risk_level = "Low"
        elif fire_risk < 0.6:
            risk_level = "Medium"
        elif fire_risk < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        return {
            'fire_risk_probability': fire_risk,
            'risk_level': risk_level,
            'confidence': min(max(fire_risk, 1 - fire_risk), 0.95)  # Cap confidence at 95%
        }

def evaluate_prediction_model(model, X_test, y_test, device='auto'):
    """Comprehensive evaluation of the prediction model"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        predictions = model(X_test).squeeze().cpu().numpy()
        targets = y_test.cpu().numpy()
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    print("🌪️ Fire Spread Prediction Model Evaluation Results")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(targets, predictions, alpha=0.6, color='blue')
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('Actual Fire Risk')
    plt.ylabel('Predicted Fire Risk')
    plt.title('Predicted vs Actual Fire Risk')
    plt.grid(True)
    
    # Residuals plot
    plt.subplot(1, 2, 2)
    residuals = targets - predictions
    plt.scatter(predictions, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Fire Risk')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'predictions': predictions,
        'targets': targets
    }

def create_fire_spread_forecast(model, current_weather, days_ahead=7, device='auto'):
    """Create multi-day fire spread forecast"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    forecasts = []
    current_sequence = current_weather.copy()
    
    model.eval()
    with torch.no_grad():
        for day in range(days_ahead):
            # Predict next day's risk
            if isinstance(current_sequence, np.ndarray):
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            else:
                input_tensor = current_sequence.unsqueeze(0).to(device)
            
            prediction = model(input_tensor)
            fire_risk = prediction.item()
            
            # Determine risk level
            if fire_risk < 0.3:
                risk_level = "Low"
            elif fire_risk < 0.6:
                risk_level = "Medium"
            elif fire_risk < 0.8:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            forecasts.append({
                'day': day + 1,
                'fire_risk_probability': fire_risk,
                'risk_level': risk_level
            })
            
            # Update sequence for next prediction (simple approach)
            # In practice, you'd want real weather forecasts
            if len(current_sequence.shape) == 2:
                current_sequence = np.roll(current_sequence, -1, axis=0)
                # Add some noise to simulate changing conditions
                current_sequence[-1] += np.random.normal(0, 0.1, current_sequence.shape[1])
                current_sequence[-1] = np.clip(current_sequence[-1], 0, 1)
    
    return forecasts 