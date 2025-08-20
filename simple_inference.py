"""
Simple inference script for trained eye state classification models
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import joblib
import config

class EyeStatePredictor:
    def __init__(self, model_name='random_forest'):
        self.model_name = model_name
        self.class_names = ['stoner', 'alcohol', 'normal']
        self.load_model()
        
    def load_model(self):
        """Load trained model and scaler"""
        model_filename = f"{self.model_name}_model.joblib"
        model_path = os.path.join(config.MODEL_DIR, model_filename)
        scaler_path = os.path.join(config.MODEL_DIR, 'feature_scaler.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded {self.model_name} model and scaler")
    
    def extract_features_from_image(self, image_path, target_size=(64, 64)):
        """Extract features from image (same as training)"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(target_size)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Extract basic features
                features = []
                
                # Color statistics for each channel
                for channel in range(3):  # RGB
                    channel_data = img_array[:, :, channel].flatten()
                    features.extend([
                        np.mean(channel_data),      # Mean
                        np.std(channel_data),       # Standard deviation
                        np.median(channel_data),    # Median
                        np.min(channel_data),       # Min
                        np.max(channel_data),       # Max
                        np.percentile(channel_data, 25),  # 25th percentile
                        np.percentile(channel_data, 75),  # 75th percentile
                    ])
                
                # Overall image statistics
                gray = np.mean(img_array, axis=2)
                features.extend([
                    np.mean(gray),              # Overall brightness
                    np.std(gray),               # Overall contrast
                    np.sum(gray > 128) / gray.size,  # Bright pixel ratio
                ])
                
                # Color ratios (useful for detecting redness)
                r_channel = img_array[:, :, 0].flatten()
                g_channel = img_array[:, :, 1].flatten()
                b_channel = img_array[:, :, 2].flatten()
                
                features.extend([
                    np.mean(r_channel) / (np.mean(g_channel) + 1e-6),  # Red/Green ratio
                    np.mean(r_channel) / (np.mean(b_channel) + 1e-6),  # Red/Blue ratio
                    np.mean(g_channel) / (np.mean(b_channel) + 1e-6),  # Green/Blue ratio
                ])
                
                # Texture features (simple)
                features.extend([
                    np.std(np.diff(gray, axis=0)),  # Vertical texture
                    np.std(np.diff(gray, axis=1)),  # Horizontal texture
                ])
                
                return np.array(features)
                
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def predict(self, image_path):
        """Predict eye state for a single image"""
        try:
            # Extract features
            features = self.extract_features_from_image(image_path)
            if features is None:
                return None, None, "Failed to extract features"
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0] if hasattr(self.model, 'predict_proba') else None
            
            predicted_class = self.class_names[prediction]
            confidence = probabilities[prediction] if probabilities is not None else None
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            return None, None, f"Prediction error: {e}"
    
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
            
            predicted_class, confidence, probabilities = self.predict(image_path)
            
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
                           for i in range(len(self.class_names))} if probabilities is not None else None
                
                results.append({
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': float(confidence) if confidence is not None else None,
                    'probabilities': prob_dict,
                    'error': None
                })
        
        return results

def test_model_on_samples():
    """Test the model on sample images from each class"""
    print("🧪 TESTING MODEL ON SAMPLE IMAGES")
    print("=" * 50)
    
    predictor = EyeStatePredictor('random_forest')
    
    for class_name in predictor.class_names:
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️  Directory not found: {class_dir}")
            continue
        
        # Get sample images
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️  No images found in {class_dir}")
            continue
        
        print(f"\n📁 Testing {class_name.upper()} samples:")
        
        # Test first 3 images
        for i, filename in enumerate(image_files[:3]):
            image_path = os.path.join(class_dir, filename)
            
            predicted_class, confidence, probabilities = predictor.predict(image_path)
            
            if predicted_class is None:
                print(f"  ❌ {filename}: {probabilities}")
                continue
            
            # Check if prediction is correct
            correct = "✅" if predicted_class == class_name else "❌"
            
            print(f"  {correct} {filename}")
            print(f"     Predicted: {predicted_class} ({confidence:.3f})")
            
            if probabilities is not None:
                print(f"     Probabilities:")
                for j, prob in enumerate(probabilities):
                    print(f"       {predictor.class_names[j]}: {prob:.3f}")

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Eye State Classification Inference')
    parser.add_argument('--model', choices=['random_forest', 'svm', 'logistic_regression'], 
                       default='random_forest', help='Model to use for prediction')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--batch', help='Path to directory containing images')
    parser.add_argument('--test', action='store_true', 
                       help='Test model on sample images from dataset')
    
    args = parser.parse_args()
    
    try:
        predictor = EyeStatePredictor(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    if args.test:
        # Test on sample images
        test_model_on_samples()
        
    elif args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        
        predicted_class, confidence, probabilities = predictor.predict(args.image)
        
        if predicted_class is None:
            print(f"❌ Prediction failed: {probabilities}")
        else:
            print(f"📷 Image: {args.image}")
            print(f"🎯 Predicted class: {predicted_class}")
            if confidence is not None:
                print(f"📊 Confidence: {confidence:.3f}")
            if probabilities is not None:
                print("📈 All probabilities:")
                for i, class_name in enumerate(predictor.class_names):
                    print(f"  {class_name}: {probabilities[i]:.3f}")
    
    elif args.batch:
        # Batch prediction
        if not os.path.exists(args.batch):
            print(f"❌ Directory not found: {args.batch}")
            sys.exit(1)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for filename in os.listdir(args.batch):
            if os.path.splitext(filename.lower())[1] in image_extensions:
                image_files.append(os.path.join(args.batch, filename))
        
        if not image_files:
            print(f"❌ No image files found in {args.batch}")
            sys.exit(1)
        
        print(f"🔍 Processing {len(image_files)} images...")
        results = predictor.predict_batch(image_files)
        
        # Print results
        correct_predictions = 0
        total_predictions = 0
        
        for result in results:
            print(f"\n📷 {result['image_path']}")
            if result['error']:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🎯 Predicted: {result['predicted_class']}")
                if result['confidence']:
                    print(f"📊 Confidence: {result['confidence']:.3f}")
                
                # Check accuracy if we can infer true class from path
                true_class = None
                for class_name in predictor.class_names:
                    if class_name in result['image_path'].lower():
                        true_class = class_name
                        break
                
                if true_class and result['predicted_class'] == true_class:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"\n📊 Batch Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    else:
        print("❌ Please specify --image, --batch, or --test")
        sys.exit(1)

if __name__ == "__main__":
    main()
