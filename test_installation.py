#!/usr/bin/env python3
"""
Test Maritime Domain Awareness System Installation

This script tests that all components can be imported and basic functionality works.
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    tests = [
        ("core.maritime_tracking_system", "Core tracking system"),
        ("preprocessing.speckle_reduction", "Speckle reduction"),
        ("preprocessing.intensity_normalization", "Intensity normalization"), 
        ("preprocessing.radiometric_calibration", "Radiometric calibration"),
    ]
    
    all_passed = True
    
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {description}")
        except Exception as e:
            print(f"❌ {description}: {e}")
            all_passed = False
    
    return all_passed

def test_class_instantiation():
    """Test that main classes can be instantiated"""
    print("\n🏗️ Testing class instantiation...")
    
    try:
        from core.maritime_tracking_system import (
            KalmanFilter, 
            ShipTracker, 
            TrajectoryAnomalyDetector
        )
        
        # Test KalmanFilter
        kf = KalmanFilter([100, 100])
        print("✅ KalmanFilter instantiation")
        
        # Test ShipTracker  
        tracker = ShipTracker()
        print("✅ ShipTracker instantiation")
        
        # Test TrajectoryAnomalyDetector
        detector = TrajectoryAnomalyDetector()
        print("✅ TrajectoryAnomalyDetector instantiation")
        
        return True
        
    except Exception as e:
        print(f"❌ Class instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that required dependencies are available"""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("torch", "PyTorch"),
        ("ultralytics", "Ultralytics YOLO"),
    ]
    
    optional_deps = [
        ("streamlit", "Streamlit (for dashboard)"),
        ("plotly", "Plotly (for dashboard)"),
        ("tensorflow", "TensorFlow (for LSTM)"),
    ]
    
    all_core_available = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
            all_core_available = False
    
    print("\n📊 Optional dependencies:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} - optional")
    
    return all_core_available

def test_file_structure():
    """Test that required files and directories exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "setup.py",
        "maritime_app.py",
        "src/__init__.py",
        "src/core/maritime_tracking_system.py",
        "src/dashboard/maritime_dashboard.py"
    ]
    
    required_dirs = [
        "src",
        "examples", 
        "tests",
        "docs",
        "scripts"
    ]
    
    all_present = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_present = False
    
    return all_present

def main():
    """Run all tests"""
    print("🚢 Maritime Domain Awareness System - Installation Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Class Instantiation", test_class_instantiation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Maritime Domain Awareness System is ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please fix the issues before deployment.")
    
    print("\n💡 Next steps:")
    if all_passed:
        print("1. 🚀 Deploy to GitHub: python deploy_to_github.py")
        print("2. 📁 Add your YOLO model to models/best.pt")
        print("3. 🧪 Run system: python maritime_app.py dashboard")
    else:
        print("1. 📦 Install missing dependencies: pip install -r requirements.txt")
        print("2. 🔧 Fix any missing files or structure issues")
        print("3. 🧪 Run this test again: python test_installation.py")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)