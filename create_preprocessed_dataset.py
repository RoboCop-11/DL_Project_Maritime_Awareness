import cv2
import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from preprocessing_functions.speckle_reduction import reduce_speckle_noise
from preprocessing_functions.intensity_normalization import process_intensity_normalization
from preprocessing_functions.radiometric_calibration import radiometric_calibration_pipeline

def preprocess_image(img, speckle_method='median', normalize=True, enhance_contrast=True, 
                    apply_radiometric=True):
    """
    Apply preprocessing steps to the image (same as Single_try.py).
    """
    processed_img = img.copy()
    
    # Step 1: Speckle noise reduction
    processed_img = reduce_speckle_noise(processed_img, method=speckle_method, kernel_size=5)
    
    # Step 2: Intensity normalization and contrast enhancement
    processed_img = process_intensity_normalization(
        processed_img, 
        normalize=normalize, 
        target_range=(0, 1),
        enhance_contrast_flag=enhance_contrast,
        contrast_method='clahe'
    )
    
    # Step 3: Radiometric calibration
    if apply_radiometric:
        processed_img = radiometric_calibration_pipeline(
            processed_img,
            scale_factor=1.1,
            offset=0,
            apply_gain=True,
            gain_value=1.2
        )
    
    return processed_img

def convert_labelme_to_yolo_segmentation(json_path, img_width, img_height, class_id=0):
    """
    Convert LabelMe polygon format to YOLO segmentation format.
    For segmentation, we keep the polygon format but normalize coordinates.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data['shapes']:
        return ""
    
    yolo_lines = []
    
    # Process all shapes (multiple ships per image possible)
    for shape in data['shapes']:
        if shape['shape_type'] != 'polygon':
            continue
            
        points = shape['points']
        
        # Normalize coordinates to [0, 1]
        normalized_points = []
        for point in points:
            x_norm = max(0, min(1, point[0] / img_width))  # Clamp to [0,1]
            y_norm = max(0, min(1, point[1] / img_height))  # Clamp to [0,1]
            normalized_points.extend([x_norm, y_norm])
        
        # Format as YOLO segmentation string (class_id + normalized polygon points)
        yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
        yolo_lines.append(yolo_line)
    
    return "\n".join(yolo_lines)

def create_preprocessed_dataset():
    """
    Create preprocessed dataset with YOLO format annotations.
    """
    input_dir = "SSDD_coco"
    output_dir = "SSDD_coco_preprocessed"
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    input_path = Path(input_dir)
    image_files = list(input_path.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                skipped_count += 1
                continue
            
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            
            # Apply preprocessing (same as Single_try.py)
            processed_img = preprocess_image(
                img, 
                speckle_method='median',
                normalize=True,
                enhance_contrast=True,
                apply_radiometric=True
            )
            
            # Save preprocessed image
            output_img_path = images_dir / img_path.name
            cv2.imwrite(str(output_img_path), processed_img)
            
            # Process corresponding JSON annotation
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                try:
                    yolo_annotation = convert_labelme_to_yolo_segmentation(json_path, img_width, img_height)
                    
                    # Save YOLO segmentation annotation
                    output_label_path = labels_dir / f"{img_path.stem}.txt"
                    with open(output_label_path, 'w') as f:
                        f.write(yolo_annotation)
                        
                except Exception as e:
                    print(f"Warning: Could not process annotation for {img_path}: {e}")
            else:
                print(f"Warning: No annotation found for {img_path}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    
    # Split dataset into train/val/test
    split_dataset(output_path)
    
    # Create dataset.yaml
    create_dataset_yaml(output_path)

def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test sets.
    """
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    image_files = [f.stem for f in image_files]  # Get filenames without extension
    
    # Shuffle the files
    np.random.seed(42)
    np.random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Move files to respective splits
    def move_files(file_list, split_name):
        for filename in file_list:
            # Move image
            src_img = images_dir / f"{filename}.jpg"
            if src_img.exists():
                dst_img = dataset_path / split_name / 'images' / src_img.name
                shutil.copy2(src_img, dst_img)
            
            # Move label
            src_label = labels_dir / f"{filename}.txt"
            if src_label.exists():
                dst_label = dataset_path / split_name / 'labels' / src_label.name
                shutil.copy2(src_label, dst_label)
    
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')
    
    print(f"Dataset split complete:")
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")

def create_dataset_yaml(dataset_path):
    """
    Create dataset.yaml file for YOLO segmentation training.
    """
    yaml_content = f"""# SSDD Ship Segmentation Dataset
path: /content/SSDD_coco_preprocessed  # dataset root dir for Colab
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test: test/images    # test images (relative to 'path')

# Classes
names:
  0: ship

# Number of classes
nc: 1

# Task type
task: segment  # segmentation task
"""
    
    yaml_path = dataset_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml for segmentation at {yaml_path}")

if __name__ == "__main__":
    print("🚀 Creating preprocessed SSDD dataset for segmentation...")
    create_preprocessed_dataset()
    print("✅ Preprocessing complete! Dataset ready for YOLO segmentation training.")