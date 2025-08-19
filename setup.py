"""
Setup script for Eye State Classification Project
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/raw/stoner",
        "data/raw/alcohol", 
        "data/raw/normal",
        "models/saved_models",
        "logs",
        "utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directories created successfully!")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check operating system
    os_name = platform.system()
    print(f"  Operating System: {os_name}")
    
    # Check available memory (approximate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"  Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM detected. Training may be slow.")
        else:
            print("✅ Sufficient RAM available")
    except ImportError:
        print("  Could not check RAM (psutil not installed)")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available. Training will use CPU (slower)")
    except ImportError:
        print("  PyTorch not installed yet - CUDA check will be available after installation")

def create_init_files():
    """Create __init__.py files for Python packages"""
    print("📝 Creating package initialization files...")
    
    packages = [
        "data_collection",
        "preprocessing", 
        "models",
        "training",
        "evaluation",
        "utils"
    ]
    
    for package in packages:
        init_file = os.path.join(package, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""\n{package.replace("_", " ").title()} module\n"""\n')
            print(f"  Created: {init_file}")
    
    print("✅ Package files created successfully!")

def run_initial_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running initial tests...")
    
    try:
        # Test imports
        import numpy as np
        import cv2
        import PIL
        print("✅ Core libraries imported successfully")
        
        # Test TensorFlow
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test configuration
        import config
        print("✅ Configuration loaded successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. 🔧 Configure settings (optional):")
    print("   Edit config.py to adjust parameters")
    print("\n2. 🚀 Run the complete pipeline:")
    print("   python main.py --step all")
    print("\n3. 📊 Or run individual steps:")
    print("   python main.py --step collect    # Collect images")
    print("   python main.py --step validate   # Validate data")
    print("   python main.py --step preprocess # Preprocess images")
    print("   python main.py --step train      # Train models")
    print("   python main.py --step evaluate   # Evaluate models")
    print("\n4. 🔍 Analyze your dataset:")
    print("   python utils/data_utils.py")
    print("\n5. 🎯 Make predictions on new images:")
    print("   python inference.py --model models/saved_models/best_tensorflow_model.h5 --image path/to/image.jpg")
    print("\n📚 Documentation:")
    print("   Check README.md for detailed instructions")
    print("\n⚠️  Important Notes:")
    print("   - Data collection requires internet connection")
    print("   - Training requires significant computational resources")
    print("   - Use GPU for faster training if available")
    print("   - Respect ethical guidelines when using this tool")

def main():
    """Main setup function"""
    print("🚀 Eye State Classification Project Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Create directories
    create_directories()
    
    # Create package files
    create_init_files()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        print("Please check the error messages above and try again")
        sys.exit(1)
    
    # Run tests
    if not run_initial_tests():
        print("\n⚠️  Setup completed but some tests failed")
        print("You may need to install additional dependencies manually")
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main()
