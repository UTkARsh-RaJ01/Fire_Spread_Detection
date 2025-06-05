"""
Fire Detection Model using CNN (ResNet-based)
Detects fire presence in satellite images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam, lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FireDetectionCNN(nn.Module):
    """CNN model for fire detection based on ResNet"""
    
    def __init__(self, num_classes=1, pretrained=True):
        super(FireDetectionCNN, self).__init__()
        
        # Use ResNet18 as backbone (beginner-friendly size)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final layer for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()  # For binary classification
        )
    
    def forward(self, x):
        return self.backbone(x)

class FireDetectionTrainer:
    """Trainer class for fire detection model"""
    
    def __init__(self, model, device='auto'):
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"🔥 Using device: {self.device}")
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            labels = labels.unsqueeze(1)  # Add dimension for BCELoss
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        return running_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                labels_reshaped = labels.unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_reshaped)
                val_loss += loss.item()
                
                # Convert to predictions
                predictions = (outputs > 0.5).float().squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        return val_loss / len(val_loader), accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader, num_epochs=20, save_dir="models/saved_models"):
        """Complete training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = save_dir / "best_fire_detection_model.pth"
        
        print("🔥 Starting Fire Detection Model Training...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, precision, recall, f1 = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss
                }, best_model_path)
                print(f"✅ New best model saved! Accuracy: {val_acc:.4f}")
            
            # Update learning rate
            self.scheduler.step()
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved at: {best_model_path}")
        
        # Save final model
        final_model_path = save_dir / "final_fire_detection_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, final_model_path)
        
        return best_model_path
    
    def plot_training_history(self, save_path="models/training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training history saved to: {save_path}")

def load_trained_model(model_path, device='auto'):
    """Load a trained fire detection model"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FireDetectionCNN()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    return model

def predict_fire_in_image(model, image_tensor, device='auto'):
    """Predict fire presence in a single image"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        output = model(image_tensor)
        probability = output.item()
        prediction = probability > 0.5
        
        return {
            'has_fire': prediction,
            'fire_probability': probability,
            'confidence': max(probability, 1 - probability)
        }

def evaluate_model(model, test_loader, device='auto'):
    """Comprehensive evaluation of the model"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probabilities = outputs.squeeze().cpu().numpy()
            predictions = (outputs > 0.5).float().squeeze().cpu().numpy()
            
            all_predictions.extend(predictions if len(predictions.shape) > 0 else [predictions])
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities if len(probabilities.shape) > 0 else [probabilities])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("🔥 Fire Detection Model Evaluation Results")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'], 
                yticklabels=['No Fire', 'Fire'])
    plt.title('Fire Detection - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    } 