# Camera capture script to detect birds using MobileNetSSD and send frames to the FastAPI backend

import numpy as np
import cv2
import os
from datetime import datetime
import time
import base64
import requests

# --- Configuration ---
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# API Endpoint
API_URL = "http://localhost:8000/process-image"

# Output directory
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Detecting birds...")

last_capture_time = time.time()
capture_interval = 2  # seconds

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detections = net.forward()

    bird_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            class_id = int(detections[0, 0, i, 1])
            class_name = classes[class_id]

            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 2)
            cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

            if class_name == "bird":
                bird_detected = True

    # Send to API if bird is detected and enough time passed
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

    # Show the video feed
    cv2.imshow("Bird Detection Feed", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
