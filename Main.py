import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Image preprocessing function
# ----------------------------
def preprocess_image(img):
    """
    Enhances image clarity and prepares it for further processing.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for contrast improvement
    enhanced = cv2.equalizeHist(gray)

    # Optionally, apply a slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Convert back to BGR to keep consistent with drawing boxes
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr

# ----------------------------
# Function to draw bounding boxes
# ----------------------------
def draw_boxes(img, boxes):
    """
    Draws visible bounding boxes on the image.
    Each box is a tuple: (x, y, w, h)
    """
    height, width = img.shape[:2]
    
    # Line thickness proportional to image size
    thickness = max(2, int(min(height, width) * 0.005))
    
    for (x, y, w, h) in boxes:
        # Draw solid rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness)  # Red boxes
        
        # Optional: draw filled transparent background for better visibility of boxes
        overlay = img.copy()
        alpha = 0.2  # transparency factor
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Optional: put box coordinates as text
        cv2.putText(img, f"{x},{y}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    max(0.5, thickness/2), (255,255,255), thickness//2)
    
    return img


# ----------------------------
# Display helper
# ----------------------------
def show_image(img, title="Image"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

# ----------------------------
# Single image pipeline
# ----------------------------
def process_single_image(image_path):
    img = cv2.imread(image_path)
    preprocessed = preprocess_image(img)

    # Example bounding boxes for testing
    boxes = [(50, 50, 100, 150), (200, 80, 120, 160)]
    img_with_boxes = draw_boxes(preprocessed.copy(), boxes)

    show_image(img_with_boxes, title="Single Image with Boxes")
    return img_with_boxes

# ----------------------------
# Batch image pipeline
# ----------------------------
def process_batch_images(folder_path):
    processed_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            preprocessed = preprocess_image(img)

            # Example boxes (you can adapt this to auto-detect objects)
            boxes = [(50, 50, 100, 150), (200, 80, 120, 160)]
            img_with_boxes = draw_boxes(preprocessed.copy(), boxes)

            processed_images.append((filename, img_with_boxes))
            # Optionally display each
            # show_image(img_with_boxes, title=filename)

    return processed_images

# ----------------------------
# Test single image (example)
# ----------------------------
single_image_path = r"C:\Users\Manan\OneDrive\Desktop\DL_Base\SSDD_coco\000001.jpg"  # replace with your test image path
process_single_image(single_image_path)

# Batch function exists but is NOT called
# batch_images = process_batch_images("your_folder_path")
