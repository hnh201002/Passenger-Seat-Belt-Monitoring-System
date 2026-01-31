from flask import Flask, render_template, request, jsonify
import base64
import os
from PIL import Image
from io import BytesIO
from Image_Yolo_HumanDetection import detect_and_save_people  # Import hàm YOLO human detection
from Yolo_Seatbelt import process_seatbelt_detection  # Import hàm seatbelt detection
from flask_cors import CORS
import uuid
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://192.168.31.220"])

# Thư mục lưu ảnh cropped và processed
CROPPED_IMAGES_DIR = r"F:\doantotnghiep\FinalProject1101\FinalProject\app\static\cropped_images"
os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)


# Function to delete all files in the cropped_images directory
def delete_old_images():
    for filename in os.listdir(CROPPED_IMAGES_DIR):
        file_path = os.path.join(CROPPED_IMAGES_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete file
        except Exception as e:
            app.logger.error(f"Error deleting file {file_path}: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture_image():
    try:
        start_time = time.time()  # Bắt đầu đo thời gian

        delete_old_images()

        data = request.get_json()
        base64_image = data.get("image")  # Get the base64 image string

        if not base64_image:
            end_time = time.time()
            total_time = end_time - start_time
            return jsonify({
                "message": "Waiting for image...",
                "cropped_images": [],
                "seatbelt_detections": [],
                "not_wearing_seatbelt": 0,
                "wearing_seatbelt": 0,
                "total_people": 0,
                "processing_time": f"{total_time:.2f} seconds"
            }), 200

        if base64_image.startswith('data:image/'):
            base64_image = base64_image.split(',')[1]

        try:
            image_data = base64.b64decode(base64_image)
            img = Image.open(BytesIO(image_data))
        except Exception as e:
            app.logger.error(f"Error decoding or opening image: {e}")
            return jsonify({"error": "Invalid image format"}), 400

        # Đổi tên tệp bằng UUID
        unique_filename = f"{uuid.uuid4().hex}_uploaded_image.jpg"
        img_path = os.path.join(CROPPED_IMAGES_DIR, unique_filename)
        img.save(img_path)

        # Call detect_and_save_people
        cropped_image_paths = detect_and_save_people(img_path, CROPPED_IMAGES_DIR)

        seatbelt_detection_paths = []
        wearing_seatbelt = 0
        not_wearing_seatbelt = 0

        for cropped_path in cropped_image_paths:
            result = process_seatbelt_detection(cropped_path, CROPPED_IMAGES_DIR)
            if result:
                processed_path, confidence = result
                seatbelt_detection_paths.append((processed_path, confidence))
                wearing_seatbelt += 1 if confidence >= 0.5 else 0
                not_wearing_seatbelt += 1 if confidence < 0.5 else 0

        end_time = time.time()  # Kết thúc đo thời gian
        total_time = end_time - start_time

        print(f"Processing time: {total_time:.2f} seconds")
        response_data = {
            "cropped_images": [f"/static/cropped_images/{os.path.basename(path)}" for path in cropped_image_paths],
            "seatbelt_detections": [{
                "path": f"/static/cropped_images/{os.path.basename(path)}",
                "confidence": int(confidence * 100)
            } for path, confidence in seatbelt_detection_paths],
            "not_wearing_seatbelt": not_wearing_seatbelt,
            "wearing_seatbelt": wearing_seatbelt,
            "total_people": len(cropped_image_paths),
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Failed to process image: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
