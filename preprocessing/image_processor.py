"""
Image preprocessing utilities for eye state classification
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
from sklearn.model_selection import train_test_split
import json
import logging
from tqdm import tqdm
import config

class ImagePreprocessor:
    def __init__(self):
        self.setup_logging()
        self.target_size = config.IMAGE_SIZE
        self.setup_augmentation()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_augmentation(self):
        """Setup augmentation pipeline"""
        self.augmentation = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=config.AUGMENTATION_PARAMS['rotation_range'], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.validation_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path, augment=False):
        """Preprocess a single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            if augment:
                transformed = self.augmentation(image=image)
            else:
                transformed = self.validation_transform(image=image)
                
            return transformed['image']
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {image_path}: {e}")
            return None
    
    def extract_eye_region(self, image_path):
        """Extract eye region from face image using OpenCV"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Load cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Take the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Extract face region
                face_roi = img[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                
                # Detect eyes in face region
                eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
                
                if len(eyes) >= 2:
                    # Take the two largest eyes
                    eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]
                    
                    # Create bounding box that includes both eyes
                    min_x = min(eye[0] for eye in eyes)
                    min_y = min(eye[1] for eye in eyes)
                    max_x = max(eye[0] + eye[2] for eye in eyes)
                    max_y = max(eye[1] + eye[3] for eye in eyes)
                    
                    # Add some padding
                    padding = 20
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(w, max_x + padding)
                    max_y = min(h, max_y + padding)
                    
                    # Extract eye region
                    eye_region = face_roi[min_y:max_y, min_x:max_x]
                    return eye_region
                elif len(eyes) == 1:
                    # If only one eye detected, extract with padding
                    x_eye, y_eye, w_eye, h_eye = eyes[0]
                    padding = 30
                    min_x = max(0, x_eye - padding)
                    min_y = max(0, y_eye - padding)
                    max_x = min(w, x_eye + w_eye + padding)
                    max_y = min(h, y_eye + h_eye + padding)
                    
                    eye_region = face_roi[min_y:max_y, min_x:max_x]
                    return eye_region
                else:
                    # No eyes detected, return upper half of face
                    return face_roi[:h//2, :]
            else:
                # No face detected, return original image
                return img
                
        except Exception as e:
            self.logger.error(f"Error extracting eye region from {image_path}: {e}")
            return None
    
    def create_dataset_splits(self, valid_images):
        """Create train/validation/test splits"""
        # Group images by class
        class_images = {}
        for img_info in valid_images:
            class_name = img_info['class']
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(img_info)
        
        train_images = []
        val_images = []
        test_images = []
        
        for class_name, images in class_images.items():
            # First split: separate test set
            train_val, test = train_test_split(
                images, 
                test_size=config.TEST_SPLIT, 
                random_state=42,
                stratify=None
            )
            
            # Second split: separate train and validation
            train, val = train_test_split(
                train_val,
                test_size=config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT),
                random_state=42,
                stratify=None
            )
            
            train_images.extend(train)
            val_images.extend(val)
            test_images.extend(test)
        
        self.logger.info(f"Dataset splits created:")
        self.logger.info(f"Train: {len(train_images)} images")
        self.logger.info(f"Validation: {len(val_images)} images")
        self.logger.info(f"Test: {len(test_images)} images")
        
        return train_images, val_images, test_images
    
    def process_and_save_dataset(self, valid_images):
        """Process all images and save to processed directory"""
        # Create splits
        train_images, val_images, test_images = self.create_dataset_splits(valid_images)
        
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        processed_data = {}
        
        for split_name, images in splits.items():
            self.logger.info(f"Processing {split_name} split...")
            
            split_dir = os.path.join(config.PROCESSED_DATA_DIR, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Create class subdirectories
            for class_name in config.CLASS_DISTRIBUTION.keys():
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
            
            processed_images = []
            
            for img_info in tqdm(images, desc=f"Processing {split_name}"):
                try:
                    # Extract eye region first
                    eye_region = self.extract_eye_region(img_info['filepath'])
                    
                    if eye_region is not None:
                        # Convert to RGB
                        eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
                        
                        # Apply preprocessing
                        augment = (split_name == 'train')  # Only augment training data
                        processed_img = self.preprocess_image_array(eye_region_rgb, augment)
                        
                        if processed_img is not None:
                            # Save processed image
                            output_filename = f"{img_info['class']}_{len(processed_images):04d}.npy"
                            output_path = os.path.join(split_dir, img_info['class'], output_filename)
                            
                            np.save(output_path, processed_img)
                            
                            processed_images.append({
                                'original_path': img_info['filepath'],
                                'processed_path': output_path,
                                'class': img_info['class'],
                                'filename': output_filename
                            })
                
                except Exception as e:
                    self.logger.error(f"Error processing {img_info['filepath']}: {e}")
                    continue
            
            processed_data[split_name] = processed_images
            self.logger.info(f"Processed {len(processed_images)} images for {split_name}")
        
        # Save metadata
        metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        return processed_data
    
    def preprocess_image_array(self, image_array, augment=False):
        """Preprocess image array (numpy array)"""
        try:
            if augment:
                transformed = self.augmentation(image=image_array)
            else:
                transformed = self.validation_transform(image=image_array)
                
            return transformed['image']
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image array: {e}")
            return None

if __name__ == "__main__":
    # Load valid images from validation step
    from data_collection.data_validator import ImageValidator
    
    validator = ImageValidator()
    valid_images, _ = validator.validate_all_classes(clean_invalid=False)
    
    # Process images
    processor = ImagePreprocessor()
    processed_data = processor.process_and_save_dataset(valid_images)
