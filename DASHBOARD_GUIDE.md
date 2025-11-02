# 🚢 Maritime Domain Awareness Dashboard - User Guide

## 🎯 Overview

The Maritime Domain Awareness Dashboard is a comprehensive web-based interface that provides real-time ship detection, tracking, and anomaly detection capabilities for SAR (Synthetic Aperture Radar) images.

## 🚀 Quick Start

### Method 1: Using the Launcher (Recommended)
```bash
python run_dashboard.py
```

### Method 2: Direct Streamlit Launch
```bash
streamlit run maritime_dashboard.py
```

The dashboard will automatically open in your web browser at `http://localhost:8501`

## 📋 Dashboard Features

### 🎛️ **Control Panel (Sidebar)**
- **System Initialization**: Initialize the YOLO model and tracking system
- **Status Monitoring**: Real-time system status indicators
- **Configuration Options**: Adjust processing parameters

### 📤 **Tab 1: Upload & Process**
Upload and process your SAR images with full preprocessing pipeline:

#### **Image Upload**
- Support for multiple image formats (JPG, JPEG, PNG)
- Batch processing capabilities
- Drag-and-drop interface

#### **Preprocessing Options**
- **Speckle Reduction**: Choose from median, Lee, or Frost filters
- **Intensity Normalization**: Enhance image contrast and brightness
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Radiometric Calibration**: Apply gain correction and calibration

#### **Processing Pipeline**
1. Image preprocessing with selected options
2. YOLO ship detection and segmentation
3. Kalman filter tracking initialization
4. Hungarian assignment for track association
5. Real-time visualization generation

### 🔍 **Tab 2: Detection Results**
View detailed detection and tracking results:

#### **Image Comparison**
- **Original Image**: Raw uploaded SAR image
- **Preprocessed Image**: Enhanced image after preprocessing
- **Detection Results**: Annotated image with bounding boxes and tracks

#### **Detection Metrics**
- Ships detected per frame
- Active track count
- Anomaly detection alerts
- Frame-by-frame analysis

#### **Track Information Table**
- Track ID and position coordinates
- Anomaly scores for each track
- Status indicators (Normal/Anomaly)

### 🛤️ **Tab 3: Tracking Analysis**
Comprehensive trajectory analysis and visualization:

#### **Interactive Trajectory Plot**
- Multi-colored ship trajectories
- Hover information for detailed track data
- Zoom and pan capabilities
- Track identification and labeling

#### **Track Statistics**
- Track length distribution histogram
- Track summary table with start/end frames
- Movement pattern analysis
- Trajectory continuity metrics

### 📊 **Tab 4: Performance Dashboard**
System performance monitoring and metrics:

#### **Key Performance Indicators**
- Total frames processed
- Detection success rate
- Tracking accuracy metrics
- Processing speed statistics

#### **Interactive Charts**
- Detection timeline over processed frames
- Active tracks visualization
- Performance trend analysis
- System resource utilization

#### **System Status Cards**
- Detection accuracy indicators
- Tracking stability metrics
- Anomaly detection performance

### 📋 **Tab 5: Reports**
Generate and download comprehensive reports:

#### **Automated Report Generation**
- Executive summary with key findings
- Technical performance metrics
- System configuration details
- Processing statistics

#### **Data Export Options**
- **Detection Data (CSV)**: Frame-by-frame detection results
- **Tracking Data (CSV)**: Complete trajectory information
- **Full Report (Markdown)**: Comprehensive system analysis

#### **System Status Dashboard**
- Real-time component status
- Health monitoring indicators
- Operational readiness checks

## 🎮 How to Use the Dashboard

### **Step 1: Initialize System**
1. Click "🚀 Initialize System" in the sidebar
2. Wait for YOLO model loading (may take a few moments)
3. Confirm "✅ System initialized successfully!" message

### **Step 2: Upload Images**
1. Go to "📤 Upload & Process" tab
2. Click "Choose SAR images" or drag files into the upload area
3. Select one or more SAR images (JPG, PNG formats)

### **Step 3: Configure Preprocessing**
1. Expand "🔧 Preprocessing Options"
2. Select desired preprocessing methods:
   - **Speckle Reduction**: Recommended "median" for most cases
   - **Intensity Normalization**: Keep enabled for better detection
   - **Contrast Enhancement**: Improves visibility in SAR images
   - **Radiometric Calibration**: Enhances detection accuracy

### **Step 4: Process Images**
1. Click "🚀 Process Images" button
2. Monitor progress bar and processing status
3. Wait for "✅ Processing complete!" confirmation

### **Step 5: Analyze Results**
1. **Detection Results**: View original, preprocessed, and annotated images
2. **Tracking Analysis**: Examine ship trajectories and movement patterns
3. **Performance Dashboard**: Monitor system performance metrics
4. **Reports**: Generate and download comprehensive analysis

## 📊 Understanding the Results

### **Detection Visualization**
- **Green Circles**: Normal ship tracks
- **Red Circles**: Anomalous ship behavior
- **Blue Rectangles**: YOLO detection bounding boxes
- **Colored Lines**: Ship trajectory paths
- **Track IDs**: Unique identifiers for each ship

### **Anomaly Detection**
- **Anomaly Score**: Higher values indicate more unusual behavior
- **Status Indicators**: 
  - ✅ Normal: Standard ship movement patterns
  - 🚨 Anomaly: Unusual movement detected

### **Performance Metrics**
- **Detection Rate**: Percentage of frames with successful ship detection
- **Track Continuity**: How well tracks are maintained across frames
- **Processing Speed**: Frames processed per second
- **System Accuracy**: Overall detection and tracking performance

## 🔧 Troubleshooting

### **Common Issues**

#### **System Won't Initialize**
- Ensure `YOLO MODELS/best.pt` exists
- Check that all preprocessing functions are available
- Verify sufficient system memory (4GB+ recommended)

#### **Images Won't Process**
- Confirm image format is supported (JPG, PNG)
- Check image file size (large images may take longer)
- Ensure sufficient disk space for processing

#### **Dashboard Won't Load**
- Verify Streamlit is installed: `pip install streamlit`
- Check port 8501 is available
- Try alternative launch: `streamlit run maritime_dashboard.py --server.port 8502`

#### **Slow Processing**
- Reduce image resolution for faster processing
- Process fewer images simultaneously
- Consider using GPU acceleration if available

### **Performance Optimization**

#### **For Better Speed**
- Use smaller image batches (5-10 images)
- Enable GPU acceleration if available
- Close other resource-intensive applications

#### **For Better Accuracy**
- Use high-quality SAR images
- Enable all preprocessing options
- Process images in sequence rather than large batches

## 📱 Dashboard Navigation Tips

### **Keyboard Shortcuts**
- `Ctrl + R`: Refresh dashboard
- `F11`: Toggle fullscreen mode
- `Ctrl + Shift + R`: Hard refresh (clear cache)

### **Browser Compatibility**
- **Recommended**: Chrome, Firefox, Edge (latest versions)
- **Mobile**: Responsive design works on tablets and phones
- **Features**: All modern browsers support full functionality

### **Data Management**
- **Session Persistence**: Results are maintained during browser session
- **Data Export**: Download results before closing browser
- **Storage**: Processed images are temporarily stored in memory

## 🎯 Best Practices

### **Image Selection**
- Use high-contrast SAR images for best detection
- Ensure ships are clearly visible in the imagery
- Process sequential frames for better tracking

### **Preprocessing Configuration**
- Start with default settings for initial testing
- Adjust speckle reduction based on image noise level
- Enable all enhancement options for maximum accuracy

### **Batch Processing**
- Process 5-10 images at a time for optimal performance
- Allow each batch to complete before starting the next
- Monitor system resources during processing

### **Result Analysis**
- Review detection results before analyzing trajectories
- Check anomaly scores for unusual patterns
- Export data regularly for offline analysis

## 🚀 Advanced Features

### **Custom Configuration**
- Modify preprocessing parameters in the dashboard
- Adjust detection confidence thresholds
- Configure tracking sensitivity settings

### **Data Integration**
- Export CSV data for external analysis
- Import results into GIS systems
- Integrate with maritime databases

### **Operational Deployment**
- Run dashboard on dedicated server
- Configure for multiple user access
- Set up automated processing workflows

## 📞 Support and Documentation

### **Additional Resources**
- `README.md`: Complete system documentation
- `PROJECT_SUMMARY.md`: Technical overview
- `maritime_report.txt`: Sample analysis report

### **System Requirements**
- **Python**: 3.7 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and processing
- **Network**: Internet connection for initial setup

---

## 🎉 Congratulations!

You now have a fully functional Maritime Domain Awareness Dashboard that provides:

✅ **Real-time ship detection** using state-of-the-art YOLO models  
✅ **Advanced multi-object tracking** with Kalman filters  
✅ **Intelligent anomaly detection** using LSTM neural networks  
✅ **Interactive visualizations** with professional dashboards  
✅ **Comprehensive reporting** with exportable data  
✅ **User-friendly interface** accessible via web browser  

**🚢 Ready for operational maritime surveillance!**