"""
Demonstration script for Maritime Domain Awareness System
Processes SSDD dataset images with YOLO detection, Kalman tracking, and LSTM anomaly detection
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from maritime_tracking_system import MaritimeTrackingSystem

def load_ssdd_images(dataset_path="SSDD_coco", max_images=100):
    """
    Load SSDD dataset images
    
    Args:
        dataset_path: path to SSDD dataset
        max_images: maximum number of images to process
        
    Returns:
        list: image paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(dataset_path, ext)
        image_paths.extend(glob.glob(pattern))
    
    # Sort by filename to ensure consistent ordering
    image_paths.sort()
    
    # Limit number of images
    if max_images > 0:
        image_paths = image_paths[:max_images]
    
    return image_paths

def simulate_temporal_sequence(image_paths, fps=1.0):
    """
    Simulate temporal sequence from static images
    This creates artificial motion by slightly shifting ship positions
    """
    sequences = []
    
    # Group images into sequences (simulate multiple ships over time)
    sequence_length = min(20, len(image_paths))
    
    for i in range(0, len(image_paths), sequence_length):
        sequence = image_paths[i:i+sequence_length]
        if len(sequence) >= 5:  # minimum sequence length
            sequences.append(sequence)
    
    return sequences

def add_artificial_motion(image, frame_idx, motion_type="normal"):
    """
    Add artificial motion to simulate ship movement
    This helps create more realistic tracking scenarios
    """
    height, width = image.shape[:2]
    
    if motion_type == "normal":
        # Small random movement
        dx = np.random.randint(-5, 6)
        dy = np.random.randint(-5, 6)
    elif motion_type == "anomalous":
        # Larger, more erratic movement
        dx = np.random.randint(-20, 21)
        dy = np.random.randint(-20, 21)
    else:
        dx, dy = 0, 0
    
    # Apply translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved_image = cv2.warpAffine(image, M, (width, height))
    
    return moved_image

def run_maritime_tracking_demo():
    """
    Run complete maritime tracking demonstration
    """
    print("🚢 Maritime Domain Awareness System Demo")
    print("=" * 50)
    
    # Initialize system
    print("Initializing tracking system...")
    system = MaritimeTrackingSystem(model_path="models/best.pt")
    
    # Load dataset
    print("Loading SSDD dataset...")
    image_paths = load_ssdd_images("SSDD_coco", max_images=50)
    print(f"Loaded {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("❌ No images found in SSDD_coco directory")
        return
    
    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    
    # Process images
    print("Processing images...")
    frame_id = 0
    
    # Process first batch for training
    training_frames = min(30, len(image_paths))
    
    for i, img_path in enumerate(tqdm(image_paths[:training_frames], desc="Training phase")):
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Add some artificial motion for more realistic tracking
        if i > 0:
            motion_type = "anomalous" if np.random.random() < 0.1 else "normal"
            image = add_artificial_motion(image, i, motion_type)
        
        # Process frame
        result = system.process_frame(image, frame_id=frame_id)
        
        # Visualize every 5th frame during training
        if i % 5 == 0:
            vis_image = system.visualize_frame(image, result)
            cv2.imwrite(f"output/visualizations/training_frame_{frame_id:04d}.jpg", vis_image)
        
        frame_id += 1
    
    # Train anomaly detector
    print("Training anomaly detector...")
    training_success = system.train_anomaly_detector()
    
    if training_success:
        print("✅ Anomaly detector trained successfully")
    else:
        print("⚠️ Anomaly detector training failed or insufficient data")
    
    # Process remaining images for testing
    print("Processing test images...")
    test_frames = image_paths[training_frames:]
    
    for i, img_path in enumerate(tqdm(test_frames, desc="Testing phase")):
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Add artificial motion (higher chance of anomalies in test)
        motion_type = "anomalous" if np.random.random() < 0.2 else "normal"
        image = add_artificial_motion(image, i, motion_type)
        
        # Process frame
        result = system.process_frame(image, frame_id=frame_id)
        
        # Visualize all test frames
        vis_image = system.visualize_frame(image, result)
        cv2.imwrite(f"output/visualizations/test_frame_{frame_id:04d}.jpg", vis_image)
        
        # Print anomaly alerts
        if result['anomalies']:
            for track_id, anomaly_info in result['anomalies'].items():
                if anomaly_info['is_anomaly']:
                    print(f"🚨 ANOMALY DETECTED - Frame {frame_id}, Track {track_id}, Score: {anomaly_info['score']:.3f}")
        
        frame_id += 1
    
    # Export results
    print("Exporting results...")
    results = system.export_results("output")
    
    # Generate summary report
    generate_summary_report(system, results)
    
    print("✅ Demo completed successfully!")
    print(f"📊 Results saved in 'output/' directory")
    print(f"🖼️ Visualizations saved in 'output/visualizations/' directory")

def generate_summary_report(system, results):
    """Generate summary report of tracking and anomaly detection results"""
    
    # Calculate statistics
    total_frames = len(system.frame_results)
    total_detections = sum(r['detections'] for r in system.frame_results)
    total_tracks = len(system.track_data)
    
    # Anomaly statistics
    anomaly_frames = 0
    total_anomalies = 0
    
    for frame_result in system.frame_results:
        if frame_result['anomalies']:
            frame_anomalies = sum(1 for a in frame_result['anomalies'].values() if a['is_anomaly'])
            if frame_anomalies > 0:
                anomaly_frames += 1
                total_anomalies += frame_anomalies
    
    # Generate report
    report = f"""
Maritime Domain Awareness System - Summary Report
================================================

Dataset Processing:
- Total frames processed: {total_frames}
- Total ship detections: {total_detections}
- Average detections per frame: {total_detections/total_frames:.2f}

Ship Tracking:
- Total unique tracks: {total_tracks}
- Average track length: {sum(len(track) for track in system.track_data.values())/total_tracks:.1f} frames

Anomaly Detection:
- Frames with anomalies: {anomaly_frames}
- Total anomalies detected: {total_anomalies}
- Anomaly rate: {(total_anomalies/total_detections)*100:.2f}% of all detections

System Performance:
- Detection model: YOLO segmentation
- Tracking algorithm: Kalman Filter + Hungarian Assignment
- Anomaly detection: LSTM Autoencoder (or Statistical fallback)

Files Generated:
- output/frame_results.csv: Frame-by-frame detection results
- output/track_data.csv: Complete tracking data
- output/anomaly_results.csv: Anomaly detection results
- output/visualizations/: Visualization images
"""
    
    # Save report
    with open("output/summary_report.txt", "w") as f:
        f.write(report)
    
    print(report)
    
    # Create visualization plots
    create_analysis_plots(system, results)

def create_analysis_plots(system, results):
    """Create analysis plots for the results"""
    
    try:
        # Plot 1: Detections over time
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        frame_ids = [r['frame_id'] for r in system.frame_results]
        detections = [r['detections'] for r in system.frame_results]
        plt.plot(frame_ids, detections, 'b-', linewidth=2)
        plt.title('Ship Detections Over Time')
        plt.xlabel('Frame ID')
        plt.ylabel('Number of Detections')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Track count over time
        plt.subplot(2, 2, 2)
        track_counts = [len(r['tracks']) for r in system.frame_results]
        plt.plot(frame_ids, track_counts, 'g-', linewidth=2)
        plt.title('Active Tracks Over Time')
        plt.xlabel('Frame ID')
        plt.ylabel('Number of Active Tracks')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Anomaly scores over time
        plt.subplot(2, 2, 3)
        anomaly_scores = []
        anomaly_frame_ids = []
        
        for frame_result in system.frame_results:
            for track_id, anomaly_info in frame_result['anomalies'].items():
                anomaly_scores.append(anomaly_info['score'])
                anomaly_frame_ids.append(frame_result['frame_id'])
        
        if anomaly_scores:
            plt.scatter(anomaly_frame_ids, anomaly_scores, c=['red' if s > 2.0 else 'blue' for s in anomaly_scores], alpha=0.6)
            plt.title('Anomaly Scores Over Time')
            plt.xlabel('Frame ID')
            plt.ylabel('Anomaly Score')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No anomaly data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Anomaly Scores Over Time')
        
        # Plot 4: Track length distribution
        plt.subplot(2, 2, 4)
        track_lengths = [len(track) for track in system.track_data.values()]
        if track_lengths:
            plt.hist(track_lengths, bins=20, alpha=0.7, color='purple')
            plt.title('Track Length Distribution')
            plt.xlabel('Track Length (frames)')
            plt.ylabel('Number of Tracks')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No track data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Track Length Distribution')
        
        plt.tight_layout()
        plt.savefig('output/analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Analysis plots saved as 'output/analysis_plots.png'")
        
    except Exception as e:
        print(f"⚠️ Could not create analysis plots: {e}")

def create_video_from_frames():
    """Create video from visualization frames"""
    try:
        import cv2
        
        # Get all visualization frames
        vis_dir = "output/visualizations"
        frame_files = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")))
        
        if len(frame_files) == 0:
            print("No visualization frames found")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('output/maritime_tracking_demo.mp4', fourcc, 2.0, (width, height))
        
        # Add frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video.write(frame)
        
        video.release()
        print("🎥 Demo video saved as 'output/maritime_tracking_demo.mp4'")
        
    except Exception as e:
        print(f"⚠️ Could not create video: {e}")

if __name__ == "__main__":
    # Run the complete demo
    run_maritime_tracking_demo()
    
    # Create video from frames
    create_video_from_frames()
    
    print("\n🎯 Maritime Domain Awareness Demo Complete!")
    print("Check the 'output/' directory for all results and visualizations.")