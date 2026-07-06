"""
Simplified Maritime Tracking Demo
Focuses on core YOLO + Kalman + Hungarian assignment functionality
"""

import sys
import os
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime

# Import our custom system
from src.core.maritime_tracking_system import MaritimeTrackingSystem

def simple_demo():
    """
    Simple demonstration of the maritime tracking system
    """
    print("🚢 Simple Maritime Tracking Demo")
    print("=" * 40)
    
    # Check if YOLO model exists
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ YOLO model not found at {model_path}")
        print("Please ensure your trained YOLO model is in the models/ directory")
        return
    
    # Check if dataset exists
    dataset_path = "SSDD_coco"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure your SSDD dataset is in the SSDD_coco directory")
        return
    
    # Initialize system
    print("Initializing maritime tracking system...")
    try:
        system = MaritimeTrackingSystem(model_path=model_path)
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Load some test images
    print("Loading test images...")
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))[:20]  # Process first 20 images
    
    if len(image_files) == 0:
        print("❌ No images found in dataset")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory
    os.makedirs("simple_output", exist_ok=True)
    
    # Process images
    print("Processing images...")
    results = []
    
    for i, img_path in enumerate(tqdm(image_files, desc="Processing")):
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Process frame
            result = system.process_frame(image, frame_id=i)
            results.append(result)
            
            # Create visualization
            vis_image = system.visualize_frame(image, result)
            
            # Save visualization
            output_path = f"simple_output/frame_{i:03d}.jpg"
            cv2.imwrite(output_path, vis_image)
            
            # Print progress info
            if result['detections'] > 0:
                print(f"Frame {i}: {result['detections']} ships detected, {len(result['tracks'])} tracks active")
                
                # Check for anomalies
                if result['anomalies']:
                    for track_id, anomaly_info in result['anomalies'].items():
                        if anomaly_info['is_anomaly']:
                            print(f"  🚨 Anomaly detected in track {track_id} (score: {anomaly_info['score']:.2f})")
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue
    
    # Train anomaly detector if we have enough data
    print("Training anomaly detector...")
    try:
        training_success = system.train_anomaly_detector()
        if training_success:
            print("✅ Anomaly detector trained")
        else:
            print("⚠️ Anomaly detector training failed - using statistical fallback")
    except Exception as e:
        print(f"⚠️ Anomaly detector error: {e}")
    
    # Export results
    print("Exporting results...")
    try:
        system.export_results("simple_output")
        print("✅ Results exported to simple_output/")
    except Exception as e:
        print(f"⚠️ Export error: {e}")
    
    # Print summary
    total_detections = sum(r['detections'] for r in results)
    total_tracks = len(system.track_data)
    
    print("\n📊 Summary:")
    print(f"  Frames processed: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Unique tracks: {total_tracks}")
    print(f"  Output saved in: simple_output/")
    
    return results

def test_individual_components():
    """
    Test individual components of the system
    """
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    # Test 1: YOLO Detection
    print("1. Testing YOLO Detection...")
    try:
        from maritime_tracking_system import YOLOShipDetector
        
        detector = YOLOShipDetector("models/best.pt")
        
        # Load a test image
        test_images = glob.glob("SSDD_coco/*.jpg")[:1]
        if test_images:
            image = cv2.imread(test_images[0])
            centers, boxes, masks = detector.detect_ships(image)
            print(f"   ✅ Detected {len(centers)} ships")
        else:
            print("   ⚠️ No test images found")
            
    except Exception as e:
        print(f"   ❌ YOLO detection failed: {e}")
    
    # Test 2: Kalman Filter
    print("2. Testing Kalman Filter...")
    try:
        from maritime_tracking_system import KalmanFilter
        
        kf = KalmanFilter([100, 100])  # Initial position
        
        # Simulate some measurements
        measurements = [[105, 102], [110, 104], [115, 106]]
        for measurement in measurements:
            predicted = kf.predict()
            updated = kf.update(measurement)
        
        print(f"   ✅ Kalman filter working - final position: {updated}")
        
    except Exception as e:
        print(f"   ❌ Kalman filter failed: {e}")
    
    # Test 3: Ship Tracker
    print("3. Testing Ship Tracker...")
    try:
        from maritime_tracking_system import ShipTracker
        
        tracker = ShipTracker()
        
        # Simulate detections over multiple frames
        detections_sequence = [
            [[100, 100], [200, 150]],  # Frame 1: 2 ships
            [[105, 102], [205, 152]],  # Frame 2: same ships moved
            [[110, 104], [210, 154]],  # Frame 3: same ships moved
        ]
        
        for detections in detections_sequence:
            tracks = tracker.update(detections)
        
        print(f"   ✅ Tracker working - {len(tracks)} active tracks")
        
    except Exception as e:
        print(f"   ❌ Ship tracker failed: {e}")
    
    # Test 4: Anomaly Detector (basic)
    print("4. Testing Anomaly Detector...")
    try:
        from maritime_tracking_system import TrajectoryAnomalyDetector
        
        detector = TrajectoryAnomalyDetector()
        
        # Add some trajectory data
        normal_trajectory = [[i, i, 1, 1] for i in range(20)]  # Normal linear movement
        for i, state in enumerate(normal_trajectory):
            detector.add_trajectory_point(0, np.array(state))
        
        # Try to train (will use statistical fallback if TensorFlow not available)
        success = detector.train()
        print(f"   ✅ Anomaly detector initialized (trained: {success})")
        
    except Exception as e:
        print(f"   ❌ Anomaly detector failed: {e}")

def check_dependencies():
    """
    Check if all required dependencies are available
    """
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    dependencies = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing.append(name)
    
    # Check optional dependencies
    optional_deps = {
        'tensorflow': 'TensorFlow (for LSTM)',
        'sklearn': 'Scikit-learn (for preprocessing)'
    }
    
    print("\nOptional dependencies:")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} - Missing (will use fallback)")
    
    if missing:
        print(f"\n❌ Missing required dependencies: {', '.join(missing)}")
        print("Please install them using: pip install <package_name>")
        return False
    else:
        print("\n✅ All required dependencies available")
        return True

if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies before running the demo")
        exit(1)
    
    # Test individual components
    test_individual_components()
    
    # Run simple demo
    print("\n" + "="*50)
    simple_demo()
    
    print("\n🎯 Simple demo completed!")
    print("Check 'simple_output/' directory for results")