"""
Main script to run the complete eye state classification pipeline
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import config

# Import project modules
from data_collection.web_scraper import EyeImageScraper
from data_collection.data_validator import ImageValidator
from preprocessing.image_processor import ImagePreprocessor
from models.cnn_model import ModelFactory
from training.trainer import PyTorchTrainer, TensorFlowTrainer, load_processed_data
from evaluation.evaluator import ModelEvaluator

class EyeClassificationPipeline:
    """Complete pipeline for eye state classification"""
    
    def __init__(self):
        self.setup_logging()
        self.logger.info("Eye Classification Pipeline initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(config.LOGS_DIR, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def step1_collect_data(self):
        """Step 1: Collect images from web sources"""
        self.logger.info("=" * 50)
        self.logger.info("STEP 1: DATA COLLECTION")
        self.logger.info("=" * 50)
        
        scraper = EyeImageScraper()
        results = scraper.collect_all_images()
        
        self.logger.info("Data collection completed!")
        for class_name, count in results.items():
            self.logger.info(f"{class_name}: {count} images collected")
        
        return results
    
    def step2_validate_data(self):
        """Step 2: Validate and clean collected images"""
        self.logger.info("=" * 50)
        self.logger.info("STEP 2: DATA VALIDATION")
        self.logger.info("=" * 50)
        
        validator = ImageValidator()
        valid_images, invalid_images = validator.validate_all_classes(clean_invalid=True)
        
        self.logger.info("Data validation completed!")
        self.logger.info(f"Valid images: {len(valid_images)}")
        self.logger.info(f"Invalid images removed: {len(invalid_images)}")
        
        # Log class distribution
        class_counts = {}
        for img in valid_images:
            class_name = img['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            self.logger.info(f"{class_name}: {count} valid images")
        
        return valid_images
    
    def step3_preprocess_data(self, valid_images):
        """Step 3: Preprocess images for training"""
        self.logger.info("=" * 50)
        self.logger.info("STEP 3: DATA PREPROCESSING")
        self.logger.info("=" * 50)
        
        processor = ImagePreprocessor()
        processed_data = processor.process_and_save_dataset(valid_images)
        
        self.logger.info("Data preprocessing completed!")
        for split_name, images in processed_data.items():
            self.logger.info(f"{split_name}: {len(images)} processed images")
        
        return processed_data
    
    def step4_train_models(self):
        """Step 4: Train classification models"""
        self.logger.info("=" * 50)
        self.logger.info("STEP 4: MODEL TRAINING")
        self.logger.info("=" * 50)
        
        # Load processed data
        train_data, val_data, test_data = load_processed_data()
        
        self.logger.info(f"Training samples: {len(train_data)}")
        self.logger.info(f"Validation samples: {len(val_data)}")
        self.logger.info(f"Test samples: {len(test_data)}")
        
        training_results = {}
        
        # Train TensorFlow model
        self.logger.info("\nTraining TensorFlow EfficientNet model...")
        try:
            tf_classifier = ModelFactory.create_tensorflow_model('efficientnet')
            tf_trainer = TensorFlowTrainer(tf_classifier)
            tf_history = tf_trainer.train(
                train_data, val_data, 
                epochs=config.EPOCHS, 
                batch_size=config.BATCH_SIZE
            )
            training_results['tensorflow'] = tf_history
            self.logger.info("TensorFlow model training completed!")
        except Exception as e:
            self.logger.error(f"TensorFlow training failed: {e}")
        
        # Train PyTorch model
        self.logger.info("\nTraining PyTorch ResNet model...")
        try:
            pytorch_model = ModelFactory.create_pytorch_model('resnet')
            pytorch_trainer = PyTorchTrainer(pytorch_model)
            pytorch_history = pytorch_trainer.train(
                train_data, val_data,
                epochs=config.EPOCHS,
                learning_rate=config.LEARNING_RATE,
                batch_size=config.BATCH_SIZE
            )
            training_results['pytorch'] = pytorch_history
            self.logger.info("PyTorch model training completed!")
        except Exception as e:
            self.logger.error(f"PyTorch training failed: {e}")
        
        return training_results
    
    def step5_evaluate_models(self):
        """Step 5: Evaluate trained models"""
        self.logger.info("=" * 50)
        self.logger.info("STEP 5: MODEL EVALUATION")
        self.logger.info("=" * 50)
        
        evaluator = ModelEvaluator()
        evaluation_results = {}
        
        # Evaluate TensorFlow model
        tf_model_path = os.path.join(config.MODEL_DIR, 'best_tensorflow_model.h5')
        if os.path.exists(tf_model_path):
            self.logger.info("Evaluating TensorFlow model...")
            try:
                tf_metrics, _, _, _ = evaluator.evaluate_model(tf_model_path, 'tensorflow')
                evaluation_results['tensorflow'] = tf_metrics
                self.logger.info(f"TensorFlow Model Accuracy: {tf_metrics['accuracy']:.4f}")
            except Exception as e:
                self.logger.error(f"TensorFlow evaluation failed: {e}")
        
        # Evaluate PyTorch model
        pytorch_model_path = os.path.join(config.MODEL_DIR, 'best_pytorch_model.pth')
        if os.path.exists(pytorch_model_path):
            self.logger.info("Evaluating PyTorch model...")
            try:
                pytorch_metrics, _, _, _ = evaluator.evaluate_model(pytorch_model_path, 'pytorch')
                evaluation_results['pytorch'] = pytorch_metrics
                self.logger.info(f"PyTorch Model Accuracy: {pytorch_metrics['accuracy']:.4f}")
            except Exception as e:
                self.logger.error(f"PyTorch evaluation failed: {e}")
        
        return evaluation_results
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from data collection to evaluation"""
        self.logger.info("Starting complete eye classification pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Collection
            collection_results = self.step1_collect_data()
            
            # Step 2: Data Validation
            valid_images = self.step2_validate_data()
            
            # Step 3: Data Preprocessing
            processed_data = self.step3_preprocess_data(valid_images)
            
            # Step 4: Model Training
            training_results = self.step4_train_models()
            
            # Step 5: Model Evaluation
            evaluation_results = self.step5_evaluate_models()
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 50)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 50)
            self.logger.info(f"Total execution time: {duration}")
            
            # Print final results
            if evaluation_results:
                self.logger.info("\nFinal Model Performance:")
                for model_type, metrics in evaluation_results.items():
                    self.logger.info(f"{model_type.upper()} Model:")
                    self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    self.logger.info(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
                    self.logger.info(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False
    
    def run_single_step(self, step_name):
        """Run a single step of the pipeline"""
        step_functions = {
            'collect': self.step1_collect_data,
            'validate': self.step2_validate_data,
            'preprocess': lambda: self.step3_preprocess_data(self.step2_validate_data()),
            'train': self.step4_train_models,
            'evaluate': self.step5_evaluate_models
        }
        
        if step_name not in step_functions:
            self.logger.error(f"Unknown step: {step_name}")
            return False
        
        try:
            result = step_functions[step_name]()
            self.logger.info(f"Step '{step_name}' completed successfully!")
            return result
        except Exception as e:
            self.logger.error(f"Step '{step_name}' failed: {e}")
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Eye State Classification Pipeline')
    parser.add_argument('--step', choices=['collect', 'validate', 'preprocess', 'train', 'evaluate', 'all'],
                       default='all', help='Pipeline step to run')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        exec(open(args.config).read())
    
    # Initialize pipeline
    pipeline = EyeClassificationPipeline()
    
    # Run specified step(s)
    if args.step == 'all':
        success = pipeline.run_complete_pipeline()
    else:
        success = pipeline.run_single_step(args.step)
    
    if success:
        print("\n✅ Pipeline execution completed successfully!")
    else:
        print("\n❌ Pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
