"""
Inference script for eye state classification
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
import config
from models.cnn_model import ModelFactory

class EyeStatePredictor:
    """Eye state prediction class"""
    
    def __init__(self, model_path, model_type='tensorflow'):
        self.model_type = model_type
        self.class_names = ['stoner', 'alcohol', 'normal']
        self.class_colors = {'stoner': 'red', 'alcohol': 'orange', 'normal': 'green'}
        
        # Setup preprocessing
        self.setup_preprocessing()
        
        # Load model
        self.load_model(model_path)
        
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline"""
        self.preprocess = A.Compose([
            A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if self.model_type == 'tensorflow':
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded TensorFlow model from {model_path}")
        elif self.model_type == 'pytorch':
            self.model = ModelFactory.create_pytorch_model('resnet')
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print(f"Loaded PyTorch model from {model_path}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_eye_region(self, image):
        """Extract eye region from image"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Take the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract face region
            face_roi = img[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            
            if len(eyes) >= 2:
                # Take the two largest eyes
                eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]
                
                # Create bounding box that includes both eyes
                min_x = min(eye[0] for eye in eyes)
                min_y = min(eye[1] for eye in eyes)
                max_x = max(eye[0] + eye[2] for eye in eyes)
                max_y = max(eye[1] + eye[3] for eye in eyes)
                
                # Add padding
                padding = 20
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(w, max_x + padding)
                max_y = min(h, max_y + padding)
                
                # Extract eye region
                eye_region = face_roi[min_y:max_y, min_x:max_x]
                return eye_region
            elif len(eyes) == 1:
                # Single eye detected
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
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Extract eye region
        eye_region = self.extract_eye_region(image)
        
        if eye_region is None:
            return None
        
        # Convert BGR to RGB
        if len(eye_region.shape) == 3:
            eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        else:
            eye_region_rgb = eye_region
        
        # Apply preprocessing
        transformed = self.preprocess(image=eye_region_rgb)
        processed_image = transformed['image']
        
        return processed_image
    
    def predict_single_image(self, image_path):
        """Predict eye state for a single image"""
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        if processed_image is None:
            return None, None, "Failed to preprocess image"
        
        # Prepare input for model
        if self.model_type == 'tensorflow':
            # Add batch dimension
            input_tensor = np.expand_dims(processed_image, axis=0)
            
            # Get prediction
            predictions = self.model.predict(input_tensor, verbose=0)
            probabilities = predictions[0]
            predicted_class_idx = np.argmax(probabilities)
            
        elif self.model_type == 'pytorch':
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(processed_image).permute(2, 0, 1).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0].numpy()
                predicted_class_idx = np.argmax(probabilities)
        
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return predicted_class, confidence, probabilities
    
    def predict_batch(self, image_paths):
        """Predict eye states for multiple images"""
        results = []
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                results.append({
                    'image_path': image_path,
                    'predicted_class': None,
                    'confidence': None,
                    'error': 'File not found'
                })
                continue
            
            try:
                predicted_class, confidence, probabilities = self.predict_single_image(image_path)
                
                if predicted_class is None:
                    results.append({
                        'image_path': image_path,
                        'predicted_class': None,
                        'confidence': None,
                        'error': probabilities  # Error message
                    })
                else:
                    # Create probability dictionary
                    prob_dict = {self.class_names[i]: float(probabilities[i]) 
                               for i in range(len(self.class_names))}
                    
                    results.append({
                        'image_path': image_path,
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': prob_dict,
                        'error': None
                    })
                    
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'predicted_class': None,
                    'confidence': None,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with original image and probabilities"""
        # Get prediction
        predicted_class, confidence, probabilities = self.predict_single_image(image_path)
        
        if predicted_class is None:
            print(f"Failed to predict for {image_path}: {probabilities}")
            return
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Extract eye region for display
        eye_region = self.extract_eye_region(image_path)
        if eye_region is not None:
            eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        else:
            eye_region_rgb = original_img_rgb
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(original_img_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Eye region
        ax2.imshow(eye_region_rgb)
        ax2.set_title('Extracted Eye Region')
        ax2.axis('off')
        
        # Prediction probabilities
        colors = [self.class_colors[class_name] for class_name in self.class_names]
        bars = ax3.bar(self.class_names, probabilities, color=colors, alpha=0.7)
        ax3.set_title(f'Prediction: {predicted_class} ({confidence:.2%})')
        ax3.set_ylabel('Probability')
        ax3.set_ylim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Eye State Classification Inference')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--model_type', choices=['tensorflow', 'pytorch'], 
                       default='tensorflow', help='Model type')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--batch', help='Path to directory containing images')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization of predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = EyeStatePredictor(args.model, args.model_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            sys.exit(1)
        
        predicted_class, confidence, probabilities = predictor.predict_single_image(args.image)
        
        if predicted_class is None:
            print(f"Prediction failed: {probabilities}")
        else:
            print(f"Image: {args.image}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print("All probabilities:")
            for i, class_name in enumerate(predictor.class_names):
                print(f"  {class_name}: {probabilities[i]:.2%}")
        
        # Create visualization if requested
        if args.visualize:
            output_path = None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                filename = os.path.splitext(os.path.basename(args.image))[0]
                output_path = os.path.join(args.output, f"{filename}_prediction.png")
            
            predictor.visualize_prediction(args.image, output_path)
    
    # Batch prediction
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"Directory not found: {args.batch}")
            sys.exit(1)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for filename in os.listdir(args.batch):
            if os.path.splitext(filename.lower())[1] in image_extensions:
                image_files.append(os.path.join(args.batch, filename))
        
        if not image_files:
            print(f"No image files found in {args.batch}")
            sys.exit(1)
        
        print(f"Processing {len(image_files)} images...")
        results = predictor.predict_batch(image_files)
        
        # Print results
        for result in results:
            print(f"\nImage: {result['image_path']}")
            if result['error']:
                print(f"Error: {result['error']}")
            else:
                print(f"Predicted class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.2%}")
        
        # Save results if output directory specified
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            import json
            results_path = os.path.join(args.output, 'batch_predictions.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
    
    else:
        print("Please specify either --image or --batch")
        sys.exit(1)

if __name__ == "__main__":
    main()
