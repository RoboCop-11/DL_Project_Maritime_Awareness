import cv2
import json
import numpy as np
from preprocessing_functions.speckle_reduction import reduce_speckle_noise
from preprocessing_functions.intensity_normalization import process_intensity_normalization
from preprocessing_functions.radiometric_calibration import radiometric_calibration_pipeline

def read_polygon_from_json(json_path):
    """
    Reads a JSON file and returns polygon points as a list of (x, y) tuples.
    Assumes the JSON has the format similar to LabelMe:
    {
        "shapes": [
            {
                "points": [[x1, y1], [x2, y2], ...]
            }
        ]
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    points = data['shapes'][0]['points']  # take the first shape
    points = [tuple(map(int, point)) for point in points]  # convert to int tuples
    return points

def preprocess_image(img, speckle_method='median', normalize=True, enhance_contrast=True, 
                    apply_radiometric=True):
    """
    Apply preprocessing steps to the image.
    
    Args:
        img: Input image
        speckle_method: Method for speckle reduction ('lee', 'frost', 'median')
        normalize: Whether to apply intensity normalization
        enhance_contrast: Whether to enhance contrast
        apply_radiometric: Whether to apply radiometric calibration
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

def overlay_polygon_on_image(image_path, json_path, output_path=None, color=(0, 255, 0), thickness=2,
                           apply_preprocessing=True, speckle_method='median'):
    """
    Reads an image and JSON, applies preprocessing, overlays the polygon on the image, and shows/saves it.
    
    Args:
        image_path: Path to input image
        json_path: Path to JSON file with polygon coordinates
        output_path: Path to save output image
        color: Color for polygon overlay
        thickness: Thickness of polygon lines
        apply_preprocessing: Whether to apply preprocessing steps
        speckle_method: Method for speckle reduction ('lee', 'frost', 'median')
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Apply preprocessing if requested
    if apply_preprocessing:
        img = preprocess_image(img, speckle_method=speckle_method)
    
    # Read polygon points
    polygon_points = read_polygon_from_json(json_path)
    
    # Convert points to numpy array for OpenCV
    pts = np.array(polygon_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw polygon
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    
    # Show the image
    cv2.imshow('Polygon Overlay', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save if output path is given
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved image with polygon to {output_path}")

# Example usage
# With preprocessing (default)
overlay_polygon_on_image(r'SSDD_coco/000001.jpg', 
                        r'SSDD_coco/000001.json', 
                        'output_image_preprocessed.jpg', 
                        apply_preprocessing=True, 
                        speckle_method='median')

# Without preprocessing
overlay_polygon_on_image(r'SSDD_coco/000001.jpg', 
                        r'SSDD_coco/000001.json', 
                        'output_image_original.jpg', 
                        apply_preprocessing=False)
