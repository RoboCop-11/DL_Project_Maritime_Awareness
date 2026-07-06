# Maritime Domain Awareness System

A comprehensive deep learning system for maritime surveillance using YOLO object detection, Kalman filtering, Hungarian assignment, and LSTM-based anomaly detection.

## 🎯 Overview

This system processes maritime SAR (Synthetic Aperture Radar) images to:

1. **Detect and Segment Ships** using trained YOLO models
2. **Track Ships** using Kalman filters and Hungarian assignment algorithm
3. **Detect Anomalies** in ship trajectories using LSTM autoencoders
4. **Provide Real-time Monitoring** for maritime domain awareness

## 📁 Project Structure

```
maritime-domain-awareness/
├── 📂 src/                          # Source code
│   ├── 📂 core/                     # Core tracking system
│   │   ├── __init__.py
│   │   └── maritime_tracking_system.py
│   ├── 📂 dashboard/                # Web dashboard
│   │   ├── __init__.py
│   │   └── maritime_dashboard.py
│   ├── 📂 preprocessing/            # Image preprocessing
│   │   ├── __init__.py
│   │   ├── speckle_reduction.py
│   │   ├── intensity_normalization.py
│   │   └── radiometric_calibration.py
│   └── __init__.py
├── 📂 examples/                     # Demo scripts
│   ├── simple_maritime_demo.py
│   ├── complete_demo.py
│   └── run_maritime_tracking.py
├── 📂 tests/                        # Test suite
│   ├── __init__.py
│   ├── test_dashboard.py
│   └── test_dashboard_simple.py
├── 📂 scripts/                      # Utility scripts
│   ├── setup_environment.py
│   ├── analyze_results.py
│   └── create_preprocessed_dataset.py
├── 📂 docs/                         # Documentation
│   ├── DASHBOARD_GUIDE.md
│   ├── PROJECT_SUMMARY.md
│   └── FINAL_SUMMARY.md
├── 📂 notebooks/                    # Jupyter notebooks
│   └── YOLO_Ship_Segmentation_Training.ipynb
├── 📂 models/                       # Model files (not in git)
│   └── best.pt                      # YOLO model
├── 📂 data/                         # Dataset (not in git)
│   └── SSDD_coco/                   # SAR ship dataset
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Dependencies
├── 📄 requirements-dev.txt          # Development dependencies
├── 📄 setup.py                      # Package setup
├── 📄 LICENSE                       # MIT license
└── 📄 .gitignore                    # Git ignore rules
```

## 🏗️ System Architecture

```
Input Images → Preprocessing → YOLO Detection → Multi-Object Tracking → Anomaly Detection → Output
     ↓              ↓              ↓                    ↓                    ↓            ↓
SAR Images → Enhancement → Ship Bounding → Kalman Filter → LSTM Analysis → Alerts/Reports
             Filtering     Boxes/Masks     Hungarian Alg.   Autoencoder
```

### Core Components

1. **YOLOShipDetector**: Ship detection and segmentation using trained YOLO model
2. **KalmanFilter**: State estimation for ship position and velocity
3. **ShipTracker**: Multi-object tracking with Hungarian assignment
4. **TrajectoryAnomalyDetector**: LSTM-based anomaly detection in ship paths
5. **MaritimeTrackingSystem**: Complete integrated system

## 📋 Requirements

### System Requirements
- Python 3.7+
- GPU recommended (CUDA support)
- 8GB+ RAM
- 2GB+ storage for models and data

### Dependencies
```bash
# Core dependencies
opencv-python>=4.5.0
numpy>=1.21.0
torch>=1.9.0
ultralytics>=8.0.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.3.0

# Optional (for enhanced LSTM functionality)
tensorflow>=2.8.0
scikit-learn>=1.0.0
```

## 🚀 Quick Start

### Setup Environment
```bash
# Option 1: Install the package
pip install -e .

# Option 2: Install dependencies directly
pip install -r requirements.txt

# For development (includes dashboard dependencies)
pip install -r requirements-dev.txt
```

### 2. Download Model and Data
1. **Model**: Place your trained YOLO model as `models/best.pt`
2. **Dataset**: Place SSDD dataset in `data/SSDD_coco/` directory
3. **Preprocessing**: The preprocessing functions are included in `src/preprocessing/`

### 3. Run the System

#### Quick Dashboard Launch
```bash
# Launch the web dashboard
python run_dashboard.py
```

#### Run Examples
```bash
# Simple demo (recommended for first run)
python examples/simple_maritime_demo.py

# Full demo with advanced features  
python examples/run_maritime_tracking.py

# Complete pipeline demonstration
python examples/complete_demo.py
```

## 📊 Dataset

The system is designed for the **SSDD (SAR Ship Detection Dataset)**:
- 1,160 SAR images
- Ship annotations in COCO format
- Preprocessed with speckle reduction, intensity normalization, and radiometric calibration

### Preprocessing Pipeline
1. **Speckle Noise Reduction**: Lee filter, Frost filter, or median filtering
2. **Intensity Normalization**: Histogram equalization and CLAHE
3. **Radiometric Calibration**: Gain correction and dark current compensation

## 🔧 System Components

### 1. Ship Detection (YOLO)
```python
from src.core.maritime_tracking_system import YOLOShipDetector

detector = YOLOShipDetector("models/best.pt")
centers, boxes, masks = detector.detect_ships(image)
```

### 2. Multi-Object Tracking
```python
from src.core.maritime_tracking_system import ShipTracker

tracker = ShipTracker(max_disappeared=10, max_distance=100)
tracks = tracker.update(detections)
```

### 3. Anomaly Detection
```python
from src.core.maritime_tracking_system import TrajectoryAnomalyDetector

anomaly_detector = TrajectoryAnomalyDetector()
anomaly_detector.train()
score, is_anomaly = anomaly_detector.detect_anomaly(trajectory)
```

### 4. Complete System
```python
from src.core.maritime_tracking_system import MaritimeTrackingSystem

system = MaritimeTrackingSystem("models/best.pt")
result = system.process_frame(image, frame_id=0)
```

## 📈 Output and Results

### Generated Files
- `frame_results.csv`: Frame-by-frame detection results
- `track_data.csv`: Complete tracking trajectories
- `anomaly_results.csv`: Anomaly detection results
- `visualizations/`: Annotated images with tracks and anomalies
- `summary_report.txt`: System performance summary

### Visualization Features
- Ship bounding boxes and segmentation masks
- Track trajectories with unique IDs
- Anomaly highlighting (red for anomalous, green for normal)
- Real-time statistics overlay

## 🎛️ Configuration

### Tracking Parameters
```python
tracker = ShipTracker(
    max_disappeared=10,    # Max frames before track deletion
    max_distance=100       # Max distance for track association
)
```

### Anomaly Detection Parameters
```python
anomaly_detector = TrajectoryAnomalyDetector(
    sequence_length=10,    # LSTM sequence length
    feature_dim=4          # State vector dimension [x, y, vx, vy]
)
```

### YOLO Detection Parameters
```python
detector = YOLOShipDetector(
    model_path="models/best.pt",
    confidence=0.5         # Detection confidence threshold
)
```

## 🔬 Technical Details

### Kalman Filter State Model
- **State Vector**: [x, y, vx, vy] (position and velocity)
- **Motion Model**: Constant velocity with process noise
- **Measurement Model**: Position observations only

### Hungarian Assignment
- Solves optimal assignment problem between detections and tracks
- Minimizes total assignment cost (Euclidean distance)
- Handles track creation, update, and deletion

### LSTM Anomaly Detection
- **Architecture**: Encoder-Decoder LSTM autoencoder
- **Training**: Unsupervised learning on normal trajectories
- **Detection**: Reconstruction error threshold
- **Fallback**: Statistical anomaly detection if TensorFlow unavailable

## 📊 Performance Metrics

The system tracks:
- Detection accuracy and recall
- Track continuity and accuracy
- Anomaly detection precision and recall
- Processing speed (FPS)
- Memory usage

## 🛠️ Troubleshooting

### Common Issues

1. **YOLO Model Not Found**
   ```
   Solution: Ensure best.pt is in models/ directory
   ```

2. **Dataset Not Found**
   ```
   Solution: Place SSDD images in data/SSDD_coco/ directory
   ```

3. **TensorFlow Import Error**
   ```
   Solution: System will use statistical fallback for anomaly detection
   ```

4. **CUDA Out of Memory**
   ```
   Solution: Reduce batch size or use CPU inference
   ```

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Multi-camera fusion
- [ ] Advanced anomaly types (speed, direction, formation)
- [ ] Web-based monitoring dashboard
- [ ] Integration with AIS (Automatic Identification System)
- [ ] Deep learning-based trajectory prediction

## 📚 References

1. YOLO: "You Only Look Once: Unified, Real-Time Object Detection"
2. Kalman Filter: "A New Approach to Linear Filtering and Prediction Problems"
3. Hungarian Algorithm: "The Hungarian Method for Assignment Problems"
4. LSTM: "Long Short-Term Memory Networks"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Maritime Domain Awareness System** - Enhancing maritime security through advanced AI surveillance.