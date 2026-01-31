import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnxruntime as ort
import cv2

# Path to your ONNX model
model_path = r'F:\doantotnghiep\best3.onnx'
confidence_threshold = 0.5  # Default confidence threshold (50%)

# Load ONNX model
session = ort.InferenceSession(model_path)

def preprocess_image_yolo(image_path, img_size=640):
    """Preprocess the image according to YOLOv8 requirements."""
    image = cv2.imread(image_path)  # Read image in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    h, w, _ = image.shape
    scale = img_size / max(h, w)  # Compute scaling factor
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a padded image to maintain aspect ratio
    padded_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image

    # Normalize the image to [0, 1]
    padded_image = padded_image.astype(np.float32) / 255.0
    # Change (H, W, C) to (C, H, W)
    img_array = np.transpose(padded_image, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, scale, pad_x, pad_y

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to filter overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes
    areas = (x2 - x1) * (y2 - y1)

    # Sort the detections by score in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append((boxes[i], scores[i]))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute the width and height of the overlap
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        # Compute the Intersection over Union (IoU)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        # Keep indices of boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def predict(image_path):
    """Run the ONNX model and return predictions."""
    img_array, scale, pad_x, pad_y = preprocess_image_yolo(image_path)
    inputs = {session.get_inputs()[0].name: img_array}
    output = session.run(None, inputs)

    # Extract predictions
    predictions = output[0][0]

    # Extract the data
    x_centers = predictions[0]
    y_centers = predictions[1]
    widths = predictions[2]
    heights = predictions[3]
    objectness_scores = predictions[4]

    # Convert center coordinates to corner coordinates
    x_mins = x_centers - widths / 2
    x_maxs = x_centers + widths / 2
    y_mins = y_centers - heights / 2
    y_maxs = y_centers + heights / 2

    # Adjust the boxes to the original image size
    x_mins = (x_mins - pad_x) / scale
    x_maxs = (x_maxs - pad_x) / scale
    y_mins = (y_mins - pad_y) / scale
    y_maxs = (y_maxs - pad_y) / scale

    # Combine boxes and scores
    boxes = list(zip(x_mins, y_mins, x_maxs, y_maxs))
    scores = objectness_scores

    return non_max_suppression(boxes, scores, iou_threshold=0.8)

def draw_boxes(image_path, predictions, save_path):
    """
    Draw the bounding box with the highest confidence score on the image, save it, and return the path.
    Args:
        image_path (str): Path to the input image.
        predictions (list): List of predictions as ((x_min, y_min, x_max, y_max), confidence_score).
        save_path (str): Path to save the resulting image.
    Returns:
        str: The path to the saved image.
    """
    if not predictions:
        print("No predictions to draw.")
        return None

    # Sort predictions by confidence score in descending order
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Keep only the highest-confidence prediction
    highest_confidence_prediction = predictions[0]

    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        # Use a TrueType font for better rendering
        font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size as needed
    except IOError:
        # Fallback to default font if specified font is unavailable
        font = ImageFont.load_default()

    # Extract the coordinates and confidence score
    (x_min, y_min, x_max, y_max), score = highest_confidence_prediction
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    # Set bounding box color based on confidence threshold
    box_color = "lime" if score >= 0.5 else "red"

    # Draw the bounding box
    draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=3)

    # Prepare text to display the confidence score
    text = f"Confidence: {int(score * 100)}%"

    # Calculate text size using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Define coordinates for the text background
    y0 = max(y_min - text_height - 4, 0)  # Ensure non-negative top coordinate
    y1 = y_min  # Bottom coordinate is always `y_min`
    x1 = x_min + text_width + 4  # Width of the text background

    # Draw the text background
    draw.rectangle([x_min, y0, x1, y1], fill="black")

    # Draw the confidence score text
    draw.text((x_min + 2, max(y_min - text_height - 2, 0)), text, fill="white", font=font)

    # Save the image with the highest confidence bounding box
    image.save(save_path)

    # Return the path to the saved image
    return save_path


def process_seatbelt_detection(image_path, save_dir):
    """
    Process an image to detect seatbelts and save the result.
    Args:
        image_path (str): Path to the input image.
        save_dir (str): Directory to save the processed image.
    Returns:
        tuple: (path_to_saved_image, confidence_score) or None if no detections.
    """
    predictions = predict(image_path)
    if not predictions:
        print("No detections found.")
        return None

    # Get the highest-confidence prediction
    highest_confidence_prediction = max(predictions, key=lambda x: x[1])  # (box, confidence)
    (x_min, y_min, x_max, y_max), confidence = highest_confidence_prediction

    # Save the image with the bounding box
    save_path = os.path.join(save_dir, f"processed_{os.path.basename(image_path)}")
    draw_boxes(image_path, [highest_confidence_prediction], save_path)

    return save_path, confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python Yolo_Seatbelt.py <input_image_path> <output_image_dir>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    result_path = process_seatbelt_detection(input_image_path, output_dir)
    if result_path:
        print(f"Processed image saved to {result_path}")
    else:
        print("No seatbelt detections found.")
