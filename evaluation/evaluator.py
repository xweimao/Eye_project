"""
Model evaluation utilities for eye state classification
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import config
from training.trainer import EyeDataset, load_processed_data
from models.cnn_model import ModelFactory

class ModelEvaluator:
    """Model evaluation class"""
    
    def __init__(self):
        self.setup_logging()
        self.class_names = ['stoner', 'alcohol', 'normal']
        self.class_to_idx = {'stoner': 0, 'alcohol': 1, 'normal': 2}
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_pytorch_model(self, model, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Evaluate PyTorch model"""
        model.to(device)
        model.eval()
        
        # Create test dataset and loader
        test_dataset = EyeDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating PyTorch model"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Get probabilities and predictions
                probabilities = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def evaluate_tensorflow_model(self, model, test_data):
        """Evaluate TensorFlow model"""
        # Prepare test data
        test_images = []
        test_labels = []
        
        for item in tqdm(test_data, desc="Loading test data"):
            image = np.load(item['processed_path'])
            test_images.append(image)
            test_labels.append(self.class_to_idx[item['class']])
        
        X_test = np.array(test_images)
        y_test = np.array(test_labels)
        
        # Get predictions
        probabilities = model.predict(X_test, verbose=1)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, y_test, probabilities
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        # AUC scores if probabilities are provided
        if y_prob is not None:
            try:
                # One-vs-rest AUC for multiclass
                auc_scores = []
                for i in range(len(self.class_names)):
                    y_true_binary = (y_true == i).astype(int)
                    y_prob_binary = y_prob[:, i]
                    auc = roc_auc_score(y_true_binary, y_prob_binary)
                    auc_scores.append(auc)
                
                metrics['auc_per_class'] = auc_scores
                metrics['auc_macro'] = np.mean(auc_scores)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC scores: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_prob, save_path=None):
        """Plot ROC curves for each class"""
        if y_prob is None:
            self.logger.warning("No probabilities provided, cannot plot ROC curves")
            return
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            # Create binary labels for current class
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_prob[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
            auc = roc_auc_score(y_true_binary, y_prob_binary)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        true_counts = np.bincount(y_true)
        ax1.bar(self.class_names, true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Class Distribution')
        ax1.set_ylabel('Count')
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred)
        ax2.bar(self.class_names, pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Class Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """Generate and save detailed classification report"""
        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4)
        
        self.logger.info("Classification Report:")
        self.logger.info("\n" + report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Eye State Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
        
        return report
    
    def evaluate_model(self, model_path, model_type='tensorflow', test_data=None):
        """Complete model evaluation pipeline"""
        if test_data is None:
            _, _, test_data = load_processed_data()
        
        self.logger.info(f"Evaluating {model_type} model: {model_path}")
        self.logger.info(f"Test samples: {len(test_data)}")
        
        # Load and evaluate model
        if model_type == 'tensorflow':
            model = tf.keras.models.load_model(model_path)
            y_pred, y_true, y_prob = self.evaluate_tensorflow_model(model, test_data)
        elif model_type == 'pytorch':
            # Load PyTorch model
            model = ModelFactory.create_pytorch_model('resnet')
            model.load_state_dict(torch.load(model_path))
            y_pred, y_true, y_prob = self.evaluate_pytorch_model(model, test_data)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Generate visualizations
        results_dir = os.path.join(config.LOGS_DIR, f'{model_type}_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        # Confusion matrix
        cm = self.plot_confusion_matrix(
            y_true, y_pred, 
            os.path.join(results_dir, 'confusion_matrix.png')
        )
        
        # ROC curves
        self.plot_roc_curves(
            y_true, y_prob,
            os.path.join(results_dir, 'roc_curves.png')
        )
        
        # Class distribution
        self.plot_class_distribution(
            y_true, y_pred,
            os.path.join(results_dir, 'class_distribution.png')
        )
        
        # Classification report
        report = self.generate_classification_report(
            y_true, y_pred,
            os.path.join(results_dir, 'classification_report.txt')
        )
        
        # Save metrics
        metrics_path = os.path.join(results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                else:
                    metrics_json[key] = value
            json.dump(metrics_json, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Results saved to: {results_dir}")
        
        return metrics, y_true, y_pred, y_prob

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Evaluate TensorFlow model
    tf_model_path = os.path.join(config.MODEL_DIR, 'best_tensorflow_model.h5')
    if os.path.exists(tf_model_path):
        tf_metrics, _, _, _ = evaluator.evaluate_model(tf_model_path, 'tensorflow')
        print(f"TensorFlow Model Accuracy: {tf_metrics['accuracy']:.4f}")
    
    # Evaluate PyTorch model
    pytorch_model_path = os.path.join(config.MODEL_DIR, 'best_pytorch_model.pth')
    if os.path.exists(pytorch_model_path):
        pytorch_metrics, _, _, _ = evaluator.evaluate_model(pytorch_model_path, 'pytorch')
        print(f"PyTorch Model Accuracy: {pytorch_metrics['accuracy']:.4f}")
