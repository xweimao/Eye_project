"""
Data validation and cleaning utilities for eye images
"""

import os
import cv2
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import config

class ImageValidator:
    def __init__(self):
        self.setup_logging()
        self.valid_extensions = config.SUPPORTED_FORMATS
        self.min_size = (50, 50)  # Minimum image dimensions
        self.max_size = (5000, 5000)  # Maximum image dimensions
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def is_valid_image(self, filepath):
        """Check if image file is valid and meets criteria"""
        try:
            # Check file extension
            _, ext = os.path.splitext(filepath.lower())
            if ext not in self.valid_extensions:
                return False, "Invalid file extension"
                
            # Check if file exists and has content
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return False, "File doesn't exist or is empty"
                
            # Try to open with PIL
            with Image.open(filepath) as img:
                # Check image dimensions
                width, height = img.size
                if width < self.min_size[0] or height < self.min_size[1]:
                    return False, f"Image too small: {width}x{height}"
                    
                if width > self.max_size[0] or height > self.max_size[1]:
                    return False, f"Image too large: {width}x{height}"
                    
                # Check if image has valid mode
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    return False, f"Invalid image mode: {img.mode}"
                    
            # Try to read with OpenCV for additional validation
            cv_img = cv2.imread(filepath)
            if cv_img is None:
                return False, "Cannot read with OpenCV"
                
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def detect_faces_eyes(self, filepath):
        """Detect if image contains faces/eyes using OpenCV"""
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False, 0, 0
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Load face and eye cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            
            has_face_or_eyes = len(faces) > 0 or len(eyes) > 0
            
            return has_face_or_eyes, len(faces), len(eyes)
            
        except Exception as e:
            self.logger.error(f"Error in face/eye detection for {filepath}: {e}")
            return False, 0, 0
    
    def calculate_image_quality(self, filepath):
        """Calculate basic image quality metrics"""
        try:
            img = cv2.imread(filepath)
            if img is None:
                return 0.0
                
            # Convert to grayscale for quality analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (higher is better)
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error calculating quality for {filepath}: {e}")
            return 0.0
    
    def validate_class_directory(self, class_name):
        """Validate all images in a class directory"""
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            self.logger.error(f"Directory doesn't exist: {class_dir}")
            return []
            
        valid_images = []
        invalid_images = []
        
        image_files = [f for f in os.listdir(class_dir) 
                      if os.path.splitext(f.lower())[1] in self.valid_extensions]
        
        self.logger.info(f"Validating {len(image_files)} images in {class_name} class")
        
        for filename in tqdm(image_files, desc=f"Validating {class_name}"):
            filepath = os.path.join(class_dir, filename)
            
            # Basic validation
            is_valid, reason = self.is_valid_image(filepath)
            
            if is_valid:
                # Additional checks
                has_face_eyes, face_count, eye_count = self.detect_faces_eyes(filepath)
                quality_score = self.calculate_image_quality(filepath)
                
                image_info = {
                    'filepath': filepath,
                    'filename': filename,
                    'class': class_name,
                    'has_face_eyes': has_face_eyes,
                    'face_count': face_count,
                    'eye_count': eye_count,
                    'quality_score': quality_score,
                    'valid': True
                }
                
                # Only keep images with reasonable quality and face/eye detection
                if quality_score > 0.1 and (has_face_eyes or eye_count > 0):
                    valid_images.append(image_info)
                else:
                    invalid_images.append({**image_info, 'valid': False, 'reason': 'Low quality or no face/eyes detected'})
            else:
                invalid_images.append({
                    'filepath': filepath,
                    'filename': filename,
                    'class': class_name,
                    'valid': False,
                    'reason': reason
                })
        
        self.logger.info(f"{class_name}: {len(valid_images)} valid, {len(invalid_images)} invalid images")
        
        return valid_images, invalid_images
    
    def clean_invalid_images(self, invalid_images):
        """Remove invalid images from filesystem"""
        removed_count = 0
        
        for img_info in invalid_images:
            try:
                if os.path.exists(img_info['filepath']):
                    os.remove(img_info['filepath'])
                    removed_count += 1
                    self.logger.info(f"Removed invalid image: {img_info['filename']}")
            except Exception as e:
                self.logger.error(f"Error removing {img_info['filepath']}: {e}")
        
        self.logger.info(f"Removed {removed_count} invalid images")
        return removed_count
    
    def validate_all_classes(self, clean_invalid=True):
        """Validate images for all classes"""
        all_valid = []
        all_invalid = []
        
        for class_name in config.CLASS_DISTRIBUTION.keys():
            valid_images, invalid_images = self.validate_class_directory(class_name)
            all_valid.extend(valid_images)
            all_invalid.extend(invalid_images)
            
            if clean_invalid:
                self.clean_invalid_images(invalid_images)
        
        self.logger.info(f"Total validation results:")
        self.logger.info(f"Valid images: {len(all_valid)}")
        self.logger.info(f"Invalid images: {len(all_invalid)}")
        
        return all_valid, all_invalid

if __name__ == "__main__":
    validator = ImageValidator()
    valid_images, invalid_images = validator.validate_all_classes(clean_invalid=True)
