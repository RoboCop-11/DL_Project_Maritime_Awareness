# 🚢 Maritime Domain Awareness System - Project Structure

## 📁 Complete Directory Layout

```
maritime-domain-awareness/
├── 📄 README.md                         # Main project documentation
├── 📄 LICENSE                           # MIT License
├── 📄 CHANGELOG.md                      # Version history
├── 📄 CONTRIBUTING.md                   # Contribution guidelines  
├── 📄 PROJECT_STRUCTURE.md              # This file
├── 📄 requirements.txt                  # Core dependencies
├── 📄 requirements-dev.txt              # Development dependencies
├── 📄 dashboard_requirements.txt        # Dashboard-specific deps
├── 📄 setup.py                         # Package installation
├── 📄 MANIFEST.in                       # Package manifest
├── 📄 .gitignore                       # Git ignore rules
│
├── 🚀 Maritime App Entry Points
│   ├── 📄 maritime_app.py              # Main application launcher
│   ├── 📄 run_dashboard.py             # Legacy dashboard launcher
│   ├── 📄 deploy_to_github.py          # GitHub deployment script
│   ├── 📄 test_installation.py         # Installation verification
│   └── 📄 dev_setup.py                 # Development setup utility
│
├── 📦 src/                             # Source Code
│   ├── 📄 __init__.py                  # Package initialization
│   ├── 📄 __main__.py                  # Module entry point
│   │
│   ├── 🧠 core/                       # Core System Components
│   │   ├── 📄 __init__.py
│   │   └── 📄 maritime_tracking_system.py  # Main tracking system
│   │       ├── 🎯 YOLOShipDetector     # YOLO-based ship detection
│   │       ├── 🔢 KalmanFilter         # State estimation
│   │       ├── 🛤️ ShipTracker          # Multi-object tracking
│   │       ├── 🚨 TrajectoryAnomalyDetector  # LSTM anomaly detection
│   │       └── 🚢 MaritimeTrackingSystem     # Complete pipeline
│   │
│   ├── 🖥️ dashboard/                   # Web Dashboard
│   │   ├── 📄 __init__.py
│   │   └── 📄 maritime_dashboard.py    # Streamlit web interface
│   │       ├── 📤 File Upload          # Image upload interface
│   │       ├── ⚙️ Preprocessing Config # Image enhancement settings
│   │       ├── 🎯 Detection Display    # Results visualization
│   │       ├── 📊 Analytics Dashboard  # Performance metrics
│   │       └── 📋 Report Generation    # Data export
│   │
│   └── 🔧 preprocessing/               # Image Preprocessing
│       ├── 📄 __init__.py
│       ├── 📄 speckle_reduction.py     # Noise reduction (Lee, Frost, Median)
│       ├── 📄 intensity_normalization.py  # Histogram equalization, CLAHE
│       └── 📄 radiometric_calibration.py  # Gain correction, dark current
│
├── 🧪 examples/                        # Demo Scripts
│   ├── 📄 simple_maritime_demo.py      # Basic functionality demo
│   ├── 📄 complete_demo.py             # Full pipeline demonstration
│   └── 📄 run_maritime_tracking.py     # Batch processing example
│
├── 🔬 tests/                          # Test Suite
│   ├── 📄 __init__.py
│   ├── 📄 test_dashboard.py            # Dashboard functionality tests
│   └── 📄 test_dashboard_simple.py     # Basic dashboard tests
│
├── 🛠️ scripts/                        # Utility Scripts
│   ├── 📄 setup_environment.py         # Environment configuration
│   ├── 📄 analyze_results.py           # Results analysis tools
│   └── 📄 create_preprocessed_dataset.py  # Dataset preparation
│
├── 📚 docs/                           # Documentation
│   ├── 📄 DASHBOARD_GUIDE.md          # Dashboard user guide
│   ├── 📄 PROJECT_SUMMARY.md          # Technical overview
│   └── 📄 FINAL_SUMMARY.md            # Implementation summary
│
├── 📓 notebooks/                       # Jupyter Notebooks
│   └── 📄 YOLO_Ship_Segmentation_Training.ipynb  # Model training
│
├── 🧠 models/                         # Model Storage (not in git)
│   ├── 📄 README.md                   # Model documentation
│   └── 🎯 best.pt                     # Trained YOLO model (user provided)
│
├── 📊 data/                           # Dataset Storage (not in git)
│   ├── 📄 README.md                   # Dataset documentation
│   └── 🗂️ SSDD_coco/                 # SAR Ship Detection Dataset
│       ├── 🖼️ *.jpg                   # SAR images (1,160 files)
│       └── 📋 *.json                  # COCO annotations
│
├── 📈 output/                         # Generated Output (not in git)
├── 📊 results/                        # Analysis Results (not in git)
├── 🖥️ dashboard_output/               # Dashboard Exports (not in git)
└── 🔍 simple_output/                  # Demo Results (not in git)
```

## 🎯 Core Components

### 🧠 Maritime Tracking System (`src/core/`)

The heart of the system with five main classes:

1. **YOLOShipDetector**
   - Uses trained YOLO segmentation model
   - Integrates preprocessing pipeline
   - Returns ship centers, bounding boxes, masks

2. **KalmanFilter** 
   - State vector: [x, y, vx, vy]
   - Constant velocity motion model
   - Handles prediction and update cycles

3. **ShipTracker**
   - Multi-object tracking with Hungarian assignment
   - Automatic track creation and deletion
   - Configurable distance thresholds

4. **TrajectoryAnomalyDetector**
   - LSTM autoencoder for trajectory analysis
   - Statistical fallback when TensorFlow unavailable
   - Unsupervised anomaly detection

5. **MaritimeTrackingSystem**
   - Complete integrated pipeline
   - Combines all components seamlessly
   - Production-ready interface

### 🔧 Preprocessing Pipeline (`src/preprocessing/`)

Three-stage enhancement for SAR images:

1. **Speckle Reduction** (`speckle_reduction.py`)
   - Lee filter for multiplicative noise
   - Frost filter for edge preservation  
   - Median filter for impulse noise

2. **Intensity Normalization** (`intensity_normalization.py`)
   - Histogram equalization for contrast
   - CLAHE for local contrast enhancement
   - Gamma correction for brightness

3. **Radiometric Calibration** (`radiometric_calibration.py`)
   - Gain correction for sensor variations
   - Dark current compensation
   - Noise floor adjustment

### 🖥️ Web Dashboard (`src/dashboard/`)

Professional Streamlit interface with:

- **📤 File Upload**: Drag-and-drop image interface
- **⚙️ Preprocessing**: Real-time image enhancement
- **🎯 Detection**: YOLO ship detection results
- **🛤️ Tracking**: Multi-object trajectory visualization
- **🚨 Anomaly Detection**: LSTM-based trajectory analysis  
- **📊 Analytics**: Performance metrics and statistics
- **📋 Reports**: CSV export and comprehensive reports

## 🚀 Entry Points

### Primary Application Launcher
```bash
python maritime_app.py dashboard    # Launch web interface
python maritime_app.py example simple  # Run basic demo
python maritime_app.py setup       # Environment setup
```

### Legacy/Alternative Launchers
```bash
python run_dashboard.py            # Direct dashboard launch
python examples/simple_maritime_demo.py  # Basic functionality test
python examples/complete_demo.py   # Full pipeline demo
```

### Development Tools
```bash
python test_installation.py        # Verify installation
python deploy_to_github.py         # Deploy to GitHub
python dev_setup.py --all          # Complete dev environment
```

## 📊 Data Flow

```
🖼️ SAR Images → 🔧 Preprocessing → 🎯 YOLO Detection → 🛤️ Kalman Tracking → 🚨 Anomaly Detection → 📋 Reports
       ↓              ↓              ↓                    ↓                    ↓              ↓
   Raw Images → Enhancement → Ship Bounding → Hungarian Assignment → LSTM Analysis → CSV/Visualizations
                Filtering     Boxes/Masks     Track Management      Trajectory Scoring  Interactive Dashboard
```

## 🔧 Configuration

### Model Configuration
- **Default Model**: `models/best.pt`
- **Confidence Threshold**: 0.5
- **IoU Threshold**: 0.45
- **Max Detections**: 1000

### Tracking Configuration  
- **Max Disappeared**: 10 frames
- **Max Distance**: 100 pixels
- **Track ID Assignment**: Hungarian algorithm

### Anomaly Detection
- **Sequence Length**: 10 frames
- **Feature Dimension**: 4 (x, y, vx, vy)
- **LSTM Architecture**: Encoder-Decoder
- **Threshold**: Dynamic based on reconstruction error

## 📦 Dependencies

### Core Requirements (`requirements.txt`)
- **Computer Vision**: OpenCV, Pillow
- **ML/DL**: PyTorch, Ultralytics YOLO, TensorFlow
- **Data Processing**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib

### Dashboard Requirements (`requirements-dev.txt`)
- **Web Interface**: Streamlit
- **Interactive Plots**: Plotly, Seaborn
- **Development**: Pytest, Black, Flake8, MyPy

## 🎯 Usage Patterns

### 1. Research & Development
```bash
# Setup environment
python dev_setup.py --setup --dirs

# Train models  
jupyter notebook notebooks/YOLO_Ship_Segmentation_Training.ipynb

# Test components
python test_installation.py
python examples/simple_maritime_demo.py
```

### 2. Production Deployment
```bash
# Deploy system
python deploy_to_github.py

# Launch dashboard
python maritime_app.py dashboard

# Process batch data
python examples/run_maritime_tracking.py
```

### 3. Integration & API
```python
# Python API usage
from src.core.maritime_tracking_system import MaritimeTrackingSystem

system = MaritimeTrackingSystem("models/best.pt")
result = system.process_frame(image, frame_id=0)
```

## 🔒 Security & Privacy

- **Data**: Images and models not included in git
- **Secrets**: No API keys or credentials in code
- **Access**: Local-only by default (dashboard on localhost)
- **Dependencies**: Pinned versions in requirements

## 🚀 Deployment Options

1. **Local Development**: Direct Python execution
2. **Docker Container**: Containerized deployment  
3. **Cloud Services**: AWS/Azure/GCP deployment
4. **Edge Computing**: Maritime surveillance hardware
5. **API Service**: RESTful web service

---

**🎉 The Maritime Domain Awareness System is production-ready with a professional structure suitable for research, development, and operational deployment!**