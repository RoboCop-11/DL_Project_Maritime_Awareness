import cv2
import numpy as np

def normalize_intensity(image, target_range=(0, 1)):
    """
    Normalize image intensity to target range.
    
    Args:
        image: Input image
        target_range: Tuple of (min, max) for target range
    """
    img = image.astype(np.float32)
    
    # Normalize to 0-1 first
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img
    
    # Scale to target range
    target_min, target_max = target_range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled

def apply_histogram_equalization(image):
    """
    Apply histogram equalization for contrast enhancement.
    """
    if len(image.shape) == 3:
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3:
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)

def enhance_contrast(image, method='clahe', **kwargs):
    """
    Enhance image contrast using specified method.
    
    Args:
        image: Input image
        method: 'histogram' or 'clahe'
        **kwargs: Additional parameters for the specific method
    """
    if method == 'histogram':
        return apply_histogram_equalization(image)
    elif method == 'clahe':
        return apply_clahe(image, **kwargs)
    else:
        raise ValueError("Method must be 'histogram' or 'clahe'")

def process_intensity_normalization(image, normalize=True, target_range=(0, 1), 
                                  enhance_contrast_flag=False, contrast_method='clahe'):
    """
    Complete intensity normalization and contrast enhancement pipeline.
    
    Args:
        image: Input image
        normalize: Whether to normalize intensity
        target_range: Target range for normalization
        enhance_contrast_flag: Whether to enhance contrast
        contrast_method: Method for contrast enhancement
    """
    processed = image.copy()
    
    if normalize:
        processed = normalize_intensity(processed, target_range)
        # Convert back to uint8 for contrast enhancement
        if target_range == (0, 1):
            processed = (processed * 255).astype(np.uint8)
        else:
            processed = processed.astype(np.uint8)
    
    if enhance_contrast_flag:
        processed = enhance_contrast(processed, method=contrast_method)
    
    return processed