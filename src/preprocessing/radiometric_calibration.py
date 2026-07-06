import numpy as np

def simulate_radiometric_calibration(image, scale_factor=1.0, offset=0.0, preserve_relative=True):
    """
    Simulate radiometric calibration by rescaling intensity values.
    
    Args:
        image: Input image
        scale_factor: Multiplicative factor for scaling
        offset: Additive offset
        preserve_relative: Whether to preserve relative differences
    """
    img = image.astype(np.float32)
    
    if preserve_relative:
        # Preserve relative differences by scaling around the mean
        img_mean = np.mean(img)
        calibrated = (img - img_mean) * scale_factor + img_mean + offset
    else:
        # Simple linear scaling
        calibrated = img * scale_factor + offset
    
    # Clip to valid range
    calibrated = np.clip(calibrated, 0, 255)
    
    return calibrated.astype(np.uint8)

def apply_gain_correction(image, gain_map=None, uniform_gain=1.2):
    """
    Apply gain correction to simulate sensor calibration.
    
    Args:
        image: Input image
        gain_map: 2D array of gain values (same size as image), or None for uniform gain
        uniform_gain: Uniform gain value if gain_map is None
    """
    img = image.astype(np.float32)
    
    if gain_map is not None:
        if len(image.shape) == 3:
            # Apply gain to each channel
            corrected = np.zeros_like(img)
            for i in range(img.shape[2]):
                corrected[:, :, i] = img[:, :, i] * gain_map
        else:
            corrected = img * gain_map
    else:
        corrected = img * uniform_gain
    
    return np.clip(corrected, 0, 255).astype(np.uint8)

def simulate_dark_current_correction(image, dark_current=5.0):
    """
    Simulate dark current correction by subtracting a constant offset.
    
    Args:
        image: Input image
        dark_current: Dark current value to subtract
    """
    img = image.astype(np.float32)
    corrected = img - dark_current
    
    return np.clip(corrected, 0, 255).astype(np.uint8)

def radiometric_calibration_pipeline(image, scale_factor=1.0, offset=0.0, 
                                   apply_gain=False, gain_value=1.2,
                                   apply_dark_correction=False, dark_current=5.0):
    """
    Complete radiometric calibration pipeline.
    
    Args:
        image: Input image
        scale_factor: Scale factor for intensity rescaling
        offset: Offset for intensity rescaling
        apply_gain: Whether to apply gain correction
        gain_value: Uniform gain value
        apply_dark_correction: Whether to apply dark current correction
        dark_current: Dark current value
    """
    processed = image.copy()
    
    # Apply dark current correction first
    if apply_dark_correction:
        processed = simulate_dark_current_correction(processed, dark_current)
    
    # Apply gain correction
    if apply_gain:
        processed = apply_gain_correction(processed, uniform_gain=gain_value)
    
    # Apply radiometric scaling
    processed = simulate_radiometric_calibration(processed, scale_factor, offset)
    
    return processed