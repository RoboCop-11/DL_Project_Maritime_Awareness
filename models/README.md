# Models Directory

This directory contains the trained models used by the Maritime Domain Awareness System.

## Required Model

### best.pt - YOLO Ship Detection Model

Place your trained YOLO model as `models/best.pt`. This model should be:

- **Architecture**: YOLOv8 segmentation model
- **Training Data**: SSDD dataset or similar SAR imagery
- **Task**: Ship detection and segmentation
- **Format**: PyTorch (.pt) format
- **Size**: Typically 50-200MB depending on model size

### Model Training

If you need to train your own model, see:
- `notebooks/YOLO_Ship_Segmentation_Training.ipynb` for training code
- YOLO documentation for advanced configuration
- The examples use Ultralytics YOLO format

### Model Performance

A well-trained model should achieve:
- **mAP@0.5**: >0.85 for ship detection
- **mAP@0.5:0.95**: >0.6 for segmentation
- **Inference Speed**: 20-50ms per image (GPU)
- **Model Size**: <100MB for deployment efficiency

## Model Variants

You can experiment with different YOLO models:

### YOLOv8 Variants
- `yolov8n.pt` - Nano (fastest, lowest accuracy)
- `yolov8s.pt` - Small (balanced speed/accuracy)
- `yolov8m.pt` - Medium (good accuracy)
- `yolov8l.pt` - Large (high accuracy)
- `yolov8x.pt` - Extra Large (highest accuracy, slower)

### Custom Models

To use a different model:

1. **Update model path** in configuration files
2. **Verify compatibility** with Ultralytics YOLO
3. **Test performance** on your dataset
4. **Update documentation** if needed

## Model Configuration

The system loads models with these default settings:

```python
# Default YOLO model configuration
model_path = "models/best.pt"
confidence_threshold = 0.5
iou_threshold = 0.45
max_detections = 1000
```

You can modify these in:
- `src/core/maritime_tracking_system.py`
- Dashboard configuration settings
- Example script parameters

## Pretrained Models

If you don't have a custom model, you can start with:

1. **YOLO COCO weights** (for general object detection)
2. **Maritime-specific models** (if available publicly)
3. **Fine-tuned models** from research papers

## Model Versioning

For production use, consider:
- **Version control** for model files (Git LFS)
- **Model registry** for experiment tracking
- **A/B testing** for model comparison
- **Rollback capability** for model updates

## Performance Optimization

### GPU Acceleration
```python
# Enable GPU inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("models/best.pt").to(device)
```

### Model Optimization
- **TensorRT**: For NVIDIA GPU optimization
- **ONNX**: For cross-platform deployment  
- **Quantization**: For mobile/edge deployment
- **Pruning**: For model compression

## Security Considerations

- **Model integrity**: Verify model checksums
- **Source verification**: Only use trusted model sources
- **Scanning**: Check for embedded malicious code
- **Access control**: Restrict model file permissions

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   FileNotFoundError: models/best.pt not found
   ```
   **Solution**: Ensure the model file exists and path is correct

2. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use smaller batch size or CPU inference

3. **Model format error**
   ```
   RuntimeError: Error loading model
   ```
   **Solution**: Verify model was trained with compatible YOLO version

### Performance Issues

- **Slow inference**: Check GPU utilization and model size
- **Poor accuracy**: Verify model was trained on similar data
- **Memory errors**: Reduce batch size or image resolution

## Model Metrics

Track these metrics for your model:

- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of correct positive identifications  
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision across confidence thresholds
- **Inference Time**: Average time per image
- **Model Size**: File size and memory usage

## Contributing

If you have a well-performing maritime detection model:

1. **Document performance** metrics and training details
2. **Provide training code** if possible
3. **Test compatibility** with the system
4. **Submit a pull request** with model details

---

**Note**: Model files are not included in version control due to size constraints. Use Git LFS or external storage for large model files.