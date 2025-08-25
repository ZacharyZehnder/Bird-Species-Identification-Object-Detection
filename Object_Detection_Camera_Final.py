import numpy as np
import cv2
import os
from datetime import datetime
import time
import base64
import requests
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image

# --- Configuration ---
model_path = 'yolov8n.pt'  # Make sure this model is downloaded
min_confidence = 0.3

# API Endpoint
API_URL = "http://192.168.1.152:8000/process-image"

# Output directory
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 model
model = YOLO(model_path)
print("YOLOv8 model loaded.")

# Start camera using Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

print("Camera opened successfully. Detecting birds...")

last_capture_time = time.time()
capture_interval = 1  # seconds

while True:
    image = picam2.capture_array()
    if image is None:
        print("Failed to grab frame. Exiting...")
        break

    results = model.predict(image, verbose=False)[0]

    bird_detected = False

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names[class_id]

        if confidence < min_confidence:
            continue

        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if class_name.lower() == "bird":
            bird_detected = True

    current_time = time.time()
    if bird_detected and (current_time - last_capture_time) >= capture_interval:
        filename = f"bird_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, image)
        print(f"Bird detected. Image saved: {filepath}")

        # Encode image as base64
        with open(filepath, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Send to FastAPI
        try:
            response = requests.post(API_URL, json={"image_base64": image_base64})
            if response.status_code == 200:
                print("API Response:", response.json())
            else:
                print(f"API Error ({response.status_code}):", response.text)
        except Exception as e:
            print("Error sending image to API:", e)

        last_capture_time = current_time

# Optional live display (disabled for headless Pi use)
# cv2.imshow("Bird Detection", image)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
