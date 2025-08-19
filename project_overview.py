"""
Project overview and status checker for Eye State Classification Project
"""

import os
import sys
import json
from datetime import datetime
import config

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_file_exists(filepath, description=""):
    """Check if file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {filepath} ({size} bytes) {description}")
        return True
    else:
        print(f"❌ {filepath} (missing) {description}")
        return False

def check_directory_exists(dirpath, description=""):
    """Check if directory exists and print status"""
    if os.path.exists(dirpath):
        files = len([f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))])
        subdirs = len([d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))])
        print(f"✅ {dirpath}/ ({files} files, {subdirs} subdirs) {description}")
        return True
    else:
        print(f"❌ {dirpath}/ (missing) {description}")
        return False

def show_project_structure():
    """Display project structure"""
    print_header("PROJECT STRUCTURE")
    
    structure = {
        "Core Files": [
            ("config.py", "Project configuration"),
            ("main.py", "Main execution script"),
            ("setup.py", "Setup and installation script"),
            ("inference.py", "Model inference script"),
            ("requirements.txt", "Python dependencies"),
            ("README.md", "Project documentation")
        ],
        "Data Collection": [
            ("data_collection/__init__.py", "Package init"),
            ("data_collection/web_scraper.py", "Web scraping module"),
            ("data_collection/data_validator.py", "Data validation module")
        ],
        "Preprocessing": [
            ("preprocessing/__init__.py", "Package init"),
            ("preprocessing/image_processor.py", "Image processing module")
        ],
        "Models": [
            ("models/__init__.py", "Package init"),
            ("models/cnn_model.py", "CNN model definitions")
        ],
        "Training": [
            ("training/__init__.py", "Package init"),
            ("training/trainer.py", "Model training module")
        ],
        "Evaluation": [
            ("evaluation/__init__.py", "Package init"),
            ("evaluation/evaluator.py", "Model evaluation module")
        ],
        "Utilities": [
            ("utils/__init__.py", "Package init"),
            ("utils/data_utils.py", "Data analysis utilities")
        ]
    }
    
    total_files = 0
    existing_files = 0
    
    for category, files in structure.items():
        print(f"\n📁 {category}:")
        for filepath, description in files:
            if check_file_exists(filepath, f"- {description}"):
                existing_files += 1
            total_files += 1
    
    print(f"\n📊 Files Status: {existing_files}/{total_files} files exist")

def show_directory_structure():
    """Display directory structure"""
    print_header("DIRECTORY STRUCTURE")
    
    directories = [
        ("data/", "Main data directory"),
        ("data/raw/", "Raw collected images"),
        ("data/raw/stoner/", "Stoner eye images"),
        ("data/raw/alcohol/", "Alcohol eye images"),
        ("data/raw/normal/", "Normal eye images"),
        ("data/processed/", "Processed images"),
        ("models/saved_models/", "Trained model files"),
        ("logs/", "Log files and results")
    ]
    
    total_dirs = 0
    existing_dirs = 0
    
    for dirpath, description in directories:
        if check_directory_exists(dirpath, f"- {description}"):
            existing_dirs += 1
        total_dirs += 1
    
    print(f"\n📊 Directories Status: {existing_dirs}/{total_dirs} directories exist")

def show_data_status():
    """Show data collection and processing status"""
    print_header("DATA STATUS")
    
    # Raw data status
    print("📥 Raw Data Collection:")
    raw_total = 0
    for class_name, target_count in config.CLASS_DISTRIBUTION.items():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
            current_count = len(files)
            raw_total += current_count
            percentage = (current_count / target_count * 100) if target_count > 0 else 0
            status = "✅" if current_count >= target_count else "⏳"
            print(f"  {status} {class_name}: {current_count}/{target_count} ({percentage:.1f}%)")
        else:
            print(f"  ❌ {class_name}: 0/{target_count} (0.0%)")
    
    target_total = sum(config.CLASS_DISTRIBUTION.values())
    overall_percentage = (raw_total / target_total * 100) if target_total > 0 else 0
    print(f"  📊 Total: {raw_total}/{target_total} ({overall_percentage:.1f}%)")
    
    # Processed data status
    print("\n🔄 Processed Data:")
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            processed_data = json.load(f)
        
        for split_name, images in processed_data.items():
            print(f"  ✅ {split_name}: {len(images)} images")
    else:
        print("  ❌ No processed data found")

def show_model_status():
    """Show trained model status"""
    print_header("MODEL STATUS")
    
    model_files = [
        ("models/saved_models/best_tensorflow_model.h5", "TensorFlow model"),
        ("models/saved_models/best_pytorch_model.pth", "PyTorch model")
    ]
    
    trained_models = 0
    for model_path, description in model_files:
        if check_file_exists(model_path, f"- {description}"):
            trained_models += 1
    
    print(f"\n📊 Models Status: {trained_models}/{len(model_files)} models trained")

def show_system_info():
    """Show system information"""
    print_header("SYSTEM INFORMATION")
    
    print(f"🐍 Python Version: {sys.version}")
    print(f"💻 Platform: {sys.platform}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check key dependencies
    print("\n📦 Key Dependencies:")
    dependencies = [
        ("numpy", "NumPy"),
        ("opencv-python", "OpenCV"),
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib")
    ]
    
    for package, name in dependencies:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (not installed)")

def show_configuration():
    """Show current configuration"""
    print_header("CONFIGURATION")
    
    print(f"🎯 Target Images: {config.TOTAL_IMAGES}")
    print(f"📊 Class Distribution:")
    for class_name, count in config.CLASS_DISTRIBUTION.items():
        percentage = (count / config.TOTAL_IMAGES * 100)
        print(f"  - {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"🖼️  Image Size: {config.IMAGE_SIZE}")
    print(f"🔄 Batch Size: {config.BATCH_SIZE}")
    print(f"📈 Epochs: {config.EPOCHS}")
    print(f"🎓 Learning Rate: {config.LEARNING_RATE}")

def show_quick_commands():
    """Show quick command reference"""
    print_header("QUICK COMMANDS")
    
    commands = [
        ("python setup.py", "🔧 Setup project"),
        ("python main.py --step all", "🚀 Run complete pipeline"),
        ("python main.py --step collect", "📥 Collect data only"),
        ("python main.py --step train", "🎓 Train models only"),
        ("python utils/data_utils.py", "📊 Analyze dataset"),
        ("python inference.py --model models/saved_models/best_tensorflow_model.h5 --image path/to/image.jpg", "🎯 Make prediction"),
        ("python project_overview.py", "📋 Show this overview")
    ]
    
    for command, description in commands:
        print(f"  {description}")
        print(f"    {command}")
        print()

def main():
    """Main function"""
    print("👁️  EYE STATE CLASSIFICATION PROJECT OVERVIEW")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    show_project_structure()
    show_directory_structure()
    show_data_status()
    show_model_status()
    show_system_info()
    show_configuration()
    show_quick_commands()
    
    print_header("PROJECT STATUS SUMMARY")
    
    # Overall status assessment
    critical_files = [
        "config.py", "main.py", "requirements.txt",
        "data_collection/web_scraper.py", "models/cnn_model.py",
        "training/trainer.py"
    ]
    
    missing_critical = [f for f in critical_files if not os.path.exists(f)]
    
    if not missing_critical:
        print("✅ All critical files are present")
        print("🚀 Project is ready to run!")
        print("\n💡 Next steps:")
        print("   1. Run 'python setup.py' to install dependencies")
        print("   2. Run 'python main.py --step all' to start the pipeline")
    else:
        print("❌ Missing critical files:")
        for f in missing_critical:
            print(f"   - {f}")
        print("\n🔧 Please ensure all files are properly created")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
