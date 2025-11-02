import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to your dataset folder (both JPEG and JSON files)
dataset_path = r"C:\Users\Manan\OneDrive\Desktop\DL_Base\SSDD_coco"

def load_random_image_and_json():
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpeg") or f.endswith(".jpg")]
    random_file = random.choice(image_files)
    json_file = random_file.replace(".jpeg", ".json").replace(".jpg", ".json")
    image_path = os.path.join(dataset_path, random_file)
    json_path = os.path.join(dataset_path, json_file)
    
    image = cv2.imread(image_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    return image, data, random_file

def preprocess_image(img):
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize for uniform display
    img = cv2.resize(img, (512, 512))
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Increase sharpness
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    # Adjust contrast and brightness
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    return img

def draw_boxes(image, data):
    img = image.copy()
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
    return img

def display_preprocessing_pipeline():
    img, data, name = load_random_image_and_json()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw boxes on original
    orig_boxed = draw_boxes(img_rgb, data)
    
    # Preprocess and draw boxes on enhanced
    enhanced = preprocess_image(img)
    enhanced_boxed = draw_boxes(enhanced, data)

    # Show both
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_boxed)
    plt.title(f"Original - {name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_boxed)
    plt.title("Enhanced + Bounding Boxes")
    plt.axis("off")

    plt.show()

# Run the pipeline for one random image
display_preprocessing_pipeline()
