import cv2
import numpy as np
from scipy.ndimage import median_filter

def lee_filter(image, window_size=5):
    """
    Apply Lee filter for speckle noise reduction.
    """
    # Convert to float for processing
    img = image.astype(np.float32)
    
    # Calculate local mean and variance
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_var = cv2.filter2D(img**2, -1, kernel) - local_mean**2
    
    # Estimate noise variance (simple approach)
    noise_var = np.var(img) * 0.1  # Assume 10% of image variance is noise
    
    # Lee filter formula
    k = local_var / (local_var + noise_var)
    filtered = local_mean + k * (img - local_mean)
    
    return np.clip(filtered, 0, 255).astype(np.uint8)

def frost_filter(image, window_size=5, damping=2.0):
    """
    Apply Frost filter for speckle noise reduction.
    """
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    filtered = np.copy(img)
    
    pad = window_size // 2
    padded = np.pad(img, pad, mode='reflect')
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+window_size, j:j+window_size]
            center_val = window[pad, pad]
            
            # Calculate weights based on distance and intensity difference
            weights = np.exp(-damping * np.abs(window - center_val) / center_val)
            filtered[i, j] = np.sum(weights * window) / np.sum(weights)
    
    return np.clip(filtered, 0, 255).astype(np.uint8)

def apply_median_filter(image, kernel_size=5):
    """
    Apply median filter for speckle noise reduction.
    """
    if len(image.shape) == 3:
        # For color images, apply to each channel
        filtered = np.zeros_like(image)
        for i in range(image.shape[2]):
            filtered[:, :, i] = median_filter(image[:, :, i], size=kernel_size)
        return filtered
    else:
        # For grayscale images
        return median_filter(image, size=kernel_size)

def reduce_speckle_noise(image, method='median', **kwargs):
    """
    Apply speckle noise reduction using specified method.
    
    Args:
        image: Input image
        method: 'lee', 'frost', or 'median'
        **kwargs: Additional parameters for the specific filter
    """
    # Convert to grayscale if color image for Lee and Frost filters
    if len(image.shape) == 3 and method in ['lee', 'frost']:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == 'lee':
            filtered_gray = lee_filter(gray, **kwargs)
        elif method == 'frost':
            filtered_gray = frost_filter(gray, **kwargs)
        # Convert back to color
        return cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)
    else:
        if method == 'lee':
            return lee_filter(image, **kwargs)
        elif method == 'frost':
            return frost_filter(image, **kwargs)
        elif method == 'median':
            return apply_median_filter(image, **kwargs)
        else:
            raise ValueError("Method must be 'lee', 'frost', or 'median'")