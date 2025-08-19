"""
Configuration file for Eye State Classification Project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved_models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Data collection settings
TOTAL_IMAGES = 2000
CLASS_DISTRIBUTION = {
    'stoner': 500,    # 1/4 of total
    'alcohol': 500,   # 1/4 of total  
    'normal': 1000    # 2/4 of total
}

# Image settings
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# Search keywords for each class
SEARCH_KEYWORDS = {
    'stoner': [
        'red eyes marijuana', 'bloodshot eyes cannabis', 'stoned eyes',
        'marijuana red eyes', 'cannabis bloodshot', 'high red eyes',
        'weed red eyes', 'smoking marijuana eyes', 'cannabis user eyes'
    ],
    'alcohol': [
        'drunk eyes', 'alcohol bloodshot eyes', 'intoxicated eyes',
        'alcoholic red eyes', 'drinking alcohol eyes', 'drunk person eyes',
        'alcohol impaired eyes', 'hangover eyes', 'alcohol abuse eyes'
    ],
    'normal': [
        'normal healthy eyes', 'clear eyes', 'healthy person eyes',
        'sober eyes', 'normal eye color', 'healthy eye appearance',
        'clear white eyes', 'normal human eyes', 'healthy eye condition'
    ]
}

# Web scraping settings
MAX_IMAGES_PER_SEARCH = 100
DOWNLOAD_DELAY = 1  # seconds between downloads
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Model training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data augmentation settings
AUGMENTATION_PARAMS = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2],
    'contrast_range': [0.8, 1.2]
}

# Model architecture settings
MODEL_PARAMS = {
    'input_shape': (*IMAGE_SIZE, IMAGE_CHANNELS),
    'num_classes': 3,
    'dropout_rate': 0.5,
    'l2_regularization': 0.001
}

# Create directories if they don't exist
DIRECTORIES = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]
for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)
    
# Create subdirectories for each class
for class_name in CLASS_DISTRIBUTION.keys():
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
