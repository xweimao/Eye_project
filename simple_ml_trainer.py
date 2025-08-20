"""
Simple machine learning trainer using scikit-learn
"""

import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import config

class SimpleMLTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.class_names = ['stoner', 'alcohol', 'normal']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def extract_features_from_image(self, image_path, target_size=(64, 64)):
        """Extract simple features from image"""
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
    
    def load_dataset(self):
        """Load and extract features from all images"""
        print("📊 Loading dataset and extracting features...")
        
        X = []  # Features
        y = []  # Labels
        image_paths = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  Directory not found: {class_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"🔍 Processing {len(image_files)} {class_name} images...")
            
            for filename in tqdm(image_files, desc=f"Extracting {class_name} features"):
                filepath = os.path.join(class_dir, filename)
                
                # Extract features
                features = self.extract_features_from_image(filepath)
                
                if features is not None:
                    X.append(features)
                    y.append(self.class_to_idx[class_name])
                    image_paths.append(filepath)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Dataset loaded: {len(X)} samples, {X.shape[1]} features")
        print(f"📊 Class distribution:")
        for class_name in self.class_names:
            count = np.sum(y == self.class_to_idx[class_name])
            print(f"  {class_name}: {count} samples")
        
        return X, y, image_paths
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        print("\n🎓 Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n🔄 Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ {model_name} trained!")
            print(f"📈 Test Accuracy: {accuracy:.3f}")
            
            # Store results
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Save model
            model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
            model_path = os.path.join(config.MODEL_DIR, model_filename)
            joblib.dump(model, model_path)
            print(f"💾 Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(config.MODEL_DIR, 'feature_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Scaler saved: {scaler_path}")
        
        return results, X_test_scaled, y_test
    
    def evaluate_models(self, results):
        """Evaluate and compare models"""
        print("\n📊 MODEL EVALUATION")
        print("=" * 50)
        
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            print(f"\n🔍 {model_name} Results:")
            print(f"Accuracy: {result['accuracy']:.3f}")
            
            # Classification report
            report = classification_report(
                result['y_test'], 
                result['y_pred'], 
                target_names=self.class_names,
                digits=3
            )
            print("Classification Report:")
            print(report)
            
            # Confusion matrix
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            print("Confusion Matrix:")
            print(cm)
            
            # Track best model
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = model_name
            
            # Save detailed results
            result_data = {
                'model_name': model_name,
                'accuracy': float(result['accuracy']),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'class_names': self.class_names
            }
            
            result_filename = f"{model_name.lower().replace(' ', '_')}_results.json"
            result_path = os.path.join(config.LOGS_DIR, result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"💾 Results saved: {result_path}")
        
        print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.3f})")
        
        return best_model, best_accuracy
    
    def create_prediction_function(self, model_name):
        """Create a prediction function for the best model"""
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        model_path = os.path.join(config.MODEL_DIR, model_filename)
        scaler_path = os.path.join(config.MODEL_DIR, 'feature_scaler.joblib')
        
        def predict_image(image_path):
            """Predict eye state for a single image"""
            try:
                # Load model and scaler
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Extract features
                features = self.extract_features_from_image(image_path)
                if features is None:
                    return None, None, "Failed to extract features"
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
                
                predicted_class = self.class_names[prediction]
                confidence = probabilities[prediction] if probabilities is not None else None
                
                return predicted_class, confidence, probabilities
                
            except Exception as e:
                return None, None, f"Prediction error: {e}"
        
        return predict_image

def main():
    """Main training function"""
    print("🤖 SIMPLE MACHINE LEARNING TRAINER")
    print("🎯 Training eye state classification models")
    print("=" * 50)
    
    trainer = SimpleMLTrainer()
    
    try:
        # Load dataset
        X, y, image_paths = trainer.load_dataset()
        
        if len(X) == 0:
            print("❌ No valid images found!")
            return
        
        # Train models
        results, X_test, y_test = trainer.train_models(X, y)
        
        # Evaluate models
        best_model, best_accuracy = trainer.evaluate_models(results)
        
        # Create prediction function
        predict_fn = trainer.create_prediction_function(best_model)
        
        # Test prediction on a sample image
        print(f"\n🎯 Testing prediction with best model ({best_model})...")
        
        # Find a sample image
        for class_name in trainer.class_names:
            class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    sample_path = os.path.join(class_dir, files[0])
                    predicted_class, confidence, probabilities = predict_fn(sample_path)
                    
                    print(f"📷 Sample: {sample_path}")
                    print(f"🎯 Predicted: {predicted_class}")
                    if confidence:
                        print(f"📊 Confidence: {confidence:.3f}")
                    if probabilities is not None:
                        print("📈 Probabilities:")
                        for i, prob in enumerate(probabilities):
                            print(f"  {trainer.class_names[i]}: {prob:.3f}")
                    break
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print("=" * 50)
        print(f"✅ Best model: {best_model}")
        print(f"📈 Best accuracy: {best_accuracy:.3f}")
        print(f"💾 Models saved in: {config.MODEL_DIR}")
        print(f"📊 Results saved in: {config.LOGS_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready for deployment!")
    else:
        print("\n💥 Training encountered errors!")
