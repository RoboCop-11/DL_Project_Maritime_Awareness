"""
Test script to verify dashboard components work correctly
"""

import sys
import os

def test_imports():
    """Test all required imports for the dashboard"""
    print("🔍 Testing Dashboard Dependencies...")
    
    required_modules = [
        ('streamlit', 'Streamlit'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib.pyplot', 'Matplotlib'),
        ('plotly.express', 'Plotly Express'),
        ('plotly.graph_objects', 'Plotly Graph Objects'),
        ('PIL', 'Pillow'),
        ('seaborn', 'Seaborn')
    ]
    
    failed_imports = []
    
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def test_system_files():
    """Test if required system files exist"""
    print("\n📁 Testing System Files...")
    
    required_files = [
        ('maritime_tracking_system.py', 'Core tracking system'),
        ('maritime_dashboard.py', 'Dashboard application'),
        ('YOLO MODELS/best.pt', 'YOLO model'),
        ('preprocessing_functions/speckle_reduction.py', 'Speckle reduction'),
        ('preprocessing_functions/intensity_normalization.py', 'Intensity normalization'),
        ('preprocessing_functions/radiometric_calibration.py', 'Radiometric calibration')
    ]
    
    missing_files = []
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {description}")
        else:
            print(f"❌ {description}: {file_path} not found")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_maritime_system():
    """Test if the maritime tracking system can be imported"""
    print("\n🚢 Testing Maritime System...")
    
    try:
        from maritime_tracking_system import MaritimeTrackingSystem, YOLOShipDetector
        print("✅ Maritime system imports successful")
        
        # Test system initialization (without loading the model)
        print("✅ System classes available")
        return True
        
    except Exception as e:
        print(f"❌ Maritime system test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functions"""
    print("\n🔧 Testing Preprocessing Functions...")
    
    try:
        from preprocessing_functions.speckle_reduction import reduce_speckle_noise
        from preprocessing_functions.intensity_normalization import process_intensity_normalization
        from preprocessing_functions.radiometric_calibration import radiometric_calibration_pipeline
        print("✅ Preprocessing functions imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Maritime Dashboard - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_imports),
        ("System Files", test_system_files),
        ("Maritime System", test_maritime_system),
        ("Preprocessing", test_preprocessing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to launch.")
        print("\n🚀 To launch the dashboard, run:")
        print("   python run_dashboard.py")
        print("   OR")
        print("   streamlit run maritime_dashboard.py")
        return True
    else:
        print("⚠️ Some tests failed. Please fix the issues before launching the dashboard.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)