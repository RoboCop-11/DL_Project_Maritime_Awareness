"""
Environment setup script for Maritime Domain Awareness System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Setting up Maritime Domain Awareness System")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        return False
    
    # Install core requirements
    print("\n📦 Installing core requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Core requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    # Try to install optional dependencies
    print("\n📦 Installing optional dependencies...")
    optional_packages = [
        "tensorflow>=2.8.0",
        "scikit-learn>=1.0.0"
    ]
    
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package.split('>=')[0]} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ {package.split('>=')[0]} installation failed (optional)")
    
    return True

def verify_installation():
    """Verify that all components are working"""
    print("\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_good = True
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_good = False
    
    # Test optional imports
    optional_imports = [
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'Scikit-learn')
    ]
    
    print("\nOptional components:")
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} (will use fallback methods)")
    
    return all_good

def check_dataset_and_model():
    """Check if dataset and model files are present"""
    print("\n📁 Checking dataset and model files...")
    
    # Check YOLO model
    model_path = "YOLO MODELS/best.pt"
    if os.path.exists(model_path):
        print(f"✅ YOLO model found: {model_path}")
    else:
        print(f"⚠️ YOLO model not found: {model_path}")
        print("   Please ensure your trained YOLO model is in the 'YOLO MODELS' directory")
    
    # Check dataset
    dataset_path = "SSDD_coco"
    if os.path.exists(dataset_path):
        image_count = len([f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Dataset found: {dataset_path} ({image_count} images)")
    else:
        print(f"⚠️ Dataset not found: {dataset_path}")
        print("   Please ensure your SSDD dataset is in the 'SSDD_coco' directory")
    
    # Check preprocessing functions
    if os.path.exists("preprocessing_functions"):
        print("✅ Preprocessing functions found")
    else:
        print("⚠️ Preprocessing functions directory not found")

def create_directories():
    """Create necessary output directories"""
    print("\n📂 Creating output directories...")
    
    directories = [
        "output",
        "output/visualizations",
        "simple_output",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def main():
    """Main setup function"""
    print("🚢 Maritime Domain Awareness System Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed during verification")
        return False
    
    # Check dataset and model
    check_dataset_and_model()
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\n🚀 You can now run the maritime tracking system:")
    print("   python simple_maritime_demo.py  # For basic demo")
    print("   python run_maritime_tracking.py  # For full demo")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)