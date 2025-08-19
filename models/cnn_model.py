"""
CNN model architecture for eye state classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config

class PyTorchEyeClassifier(nn.Module):
    """PyTorch CNN model for eye state classification"""
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(PyTorchEyeClassifier, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class PyTorchResNetClassifier(nn.Module):
    """ResNet-based classifier using transfer learning"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(PyTorchResNetClassifier, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Modify the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class TensorFlowEyeClassifier:
    """TensorFlow/Keras CNN model for eye state classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_custom_cnn(self):
        """Build custom CNN architecture"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Feature extraction blocks
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name='eye_classifier')
        return self.model
    
    def build_transfer_learning_model(self, base_model_name='efficientnet'):
        """Build model using transfer learning"""
        
        if base_model_name == 'efficientnet':
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'resnet':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'mobilenet':
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name=f'{base_model_name}_eye_classifier')
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_custom_cnn() or build_transfer_learning_model() first.")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not built yet."
        return self.model.summary()

class ModelFactory:
    """Factory class to create different model architectures"""
    
    @staticmethod
    def create_pytorch_model(model_type='custom', **kwargs):
        """Create PyTorch model"""
        if model_type == 'custom':
            return PyTorchEyeClassifier(**kwargs)
        elif model_type == 'resnet':
            return PyTorchResNetClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported PyTorch model type: {model_type}")
    
    @staticmethod
    def create_tensorflow_model(model_type='custom', **kwargs):
        """Create TensorFlow model"""
        classifier = TensorFlowEyeClassifier(**kwargs)
        
        if model_type == 'custom':
            model = classifier.build_custom_cnn()
        elif model_type in ['efficientnet', 'resnet', 'mobilenet']:
            model = classifier.build_transfer_learning_model(model_type)
        else:
            raise ValueError(f"Unsupported TensorFlow model type: {model_type}")
        
        classifier.compile_model()
        return classifier

def get_model_config():
    """Get model configuration from config file"""
    return {
        'input_shape': config.MODEL_PARAMS['input_shape'],
        'num_classes': config.MODEL_PARAMS['num_classes'],
        'dropout_rate': config.MODEL_PARAMS['dropout_rate']
    }

if __name__ == "__main__":
    # Test model creation
    model_config = get_model_config()
    
    # Create TensorFlow model
    print("Creating TensorFlow model...")
    tf_classifier = ModelFactory.create_tensorflow_model('custom', **model_config)
    print("TensorFlow model created successfully!")
    print(tf_classifier.get_model_summary())
    
    # Create PyTorch model
    print("\nCreating PyTorch model...")
    pytorch_model = ModelFactory.create_pytorch_model('custom', **model_config)
    print("PyTorch model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in pytorch_model.parameters())}")
