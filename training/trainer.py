"""
Training script for eye state classification models
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import config
from models.cnn_model import ModelFactory

class EyeDataset(Dataset):
    """PyTorch Dataset for eye images"""
    
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.class_to_idx = {'stoner': 0, 'alcohol': 1, 'normal': 2}
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load preprocessed image
        image = np.load(item['processed_path'])
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.FloatTensor(image)
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)  # HWC to CHW
        
        # Get label
        label = self.class_to_idx[item['class']]
        
        return image, label

class PyTorchTrainer:
    """PyTorch model trainer"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_data_loaders(self, train_data, val_data, batch_size=32):
        """Create PyTorch data loaders"""
        train_dataset = EyeDataset(train_data)
        val_dataset = EyeDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_data, val_data, epochs=50, learning_rate=0.001, batch_size=32):
        """Full training loop"""
        train_loader, val_loader = self.create_data_loaders(train_data, val_data, batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(config.MODEL_DIR, 'best_pytorch_model.pth'))
                self.logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }

class TensorFlowTrainer:
    """TensorFlow model trainer"""
    
    def __init__(self, model_classifier):
        self.model_classifier = model_classifier
        self.model = model_classifier.model
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, train_data, val_data):
        """Prepare data for TensorFlow training"""
        def load_and_preprocess(data_list):
            images = []
            labels = []
            class_to_idx = {'stoner': 0, 'alcohol': 1, 'normal': 2}
            
            for item in tqdm(data_list, desc="Loading data"):
                image = np.load(item['processed_path'])
                images.append(image)
                labels.append(class_to_idx[item['class']])
            
            images = np.array(images)
            labels = keras.utils.to_categorical(labels, num_classes=3)
            
            return images, labels
        
        X_train, y_train = load_and_preprocess(train_data)
        X_val, y_val = load_and_preprocess(val_data)
        
        return X_train, y_train, X_val, y_val
    
    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """Train TensorFlow model"""
        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data(train_data, val_data)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(config.MODEL_DIR, 'best_tensorflow_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history

def load_processed_data():
    """Load processed dataset metadata"""
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    return data['train'], data['val'], data['test']

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if isinstance(history, dict):  # PyTorch format
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax2.plot(history['train_accs'], label='Train Accuracy')
        ax2.plot(history['val_accs'], label='Validation Accuracy')
    else:  # TensorFlow format
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax2.plot(history.history['accuracy'], label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load data
    train_data, val_data, test_data = load_processed_data()
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Train TensorFlow model
    print("\nTraining TensorFlow model...")
    tf_classifier = ModelFactory.create_tensorflow_model('efficientnet')
    tf_trainer = TensorFlowTrainer(tf_classifier)
    tf_history = tf_trainer.train(train_data, val_data, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    
    # Plot and save results
    plot_training_history(tf_history, os.path.join(config.LOGS_DIR, 'tensorflow_training_history.png'))
