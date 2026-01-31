import cv2
import torch
from PIL import Image
import os
import numpy as np

# Load the YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("YOLOv5 model loaded successfully")

def compute_iou(boxA, boxes):
    boxA = np.array(boxA)
    boxes = np.array(boxes)
    xA = np.maximum(boxA[0], boxes[:, 0])
    yA = np.maximum(boxA[1], boxes[:, 1])
    xB = np.minimum(boxA[2], boxes[:, 2])
    yB = np.minimum(boxA[3], boxes[:, 3])

    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def non_max_suppression(boxes, confidences, iou_threshold=0.3):
    idxs = np.argsort(confidences)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        currentBox = np.array(boxes[current])
        restBoxes = np.array([boxes[i] for i in rest])
        ious = compute_iou(currentBox, restBoxes)
        idxs = rest[ious < iou_threshold]

    return keep


def detect_and_save_people(image_path, save_dir):
    """
    Detects people in the image, crops, and saves them to the specified directory.
    Args:
        image_path (str): Path to the input image.
        save_dir (str): Path to the directory where cropped images will be saved.
    Returns:
        List[str]: List of file paths to the saved cropped images.
    """
    print(f"Processing image: {image_path}")
    print(f"Saving cropped images to: {save_dir}")

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Perform detection
    print("Running YOLOv5 detection...")
    results = model(img)
    print("Detection completed")

    # Extract bounding boxes, confidence scores, and labels
    boxes = [(int(det[0]), int(det[1]), int(det[2]), int(det[3])) for det in results.xyxy[0] if
             det[5] == 0 and det[4] > 0.4]
    confidences = [float(det[4]) for det in results.xyxy[0] if det[5] == 0 and det[4] > 0.4]

    print(f"Detected boxes: {boxes}")
    if not boxes:
        print("No detections found in the image.")
        return []

    # Apply Non-Max Suppression
    keep = non_max_suppression(boxes, confidences, iou_threshold=0.3)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Crop and save images
    kept_images = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        crop_img = img[y1:y2, x1:x2]
        crop_img_pil = Image.fromarray(crop_img)
        file_path = os.path.join(save_dir, f"person_{index + 1}.jpg")
        crop_img_pil.save(file_path)
        kept_images.append(file_path)
        print(f"Saved cropped image: {file_path}")

    return kept_images

