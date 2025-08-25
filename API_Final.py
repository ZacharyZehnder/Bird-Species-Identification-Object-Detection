import base64
import io
import os
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Firebase Setup
import firebase_admin
from firebase_admin import credentials, firestore

# Gemini Setup
from google.generativeai import GenerativeModel
import google.generativeai as genai

from ultralytics import YOLO  # YOLOv8 import

# --- Initialization ---
SERVICE_ACCOUNT_KEY_PATH = "firebase.json"
FIREBASE_PROJECT_ID = "human-detection-d91e1"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Firebase init
db = None
if os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized.")
else:
    print("Firebase service account not found. Firestore will not be used.")

# Gemini init
try:
    genai.configure(api_key="AIzaSyByu5xTmX99eJzrGus4APii8Mgk2mkq8XU")
    gemini_model = GenerativeModel("models/gemini-1.5-pro")
    print("Gemini initialized.")
except Exception as e:
    gemini_model = None
    print(f"Failed to initialize Gemini: {e}")

# YOLOv8 model
model = YOLO("models/yolov8n.pt")

# --- FastAPI App ---
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with ["http://localhost:3000"] for more control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


# --- Models ---
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    bird_species: Optional[str] = None
    image_url: Optional[str] = None

class DetectionRequest(BaseModel):
    image_base64: str

class DetectionResponse(BaseModel):
    timestamp: str
    detections: List[Detection]


# --- Endpoints ---


@app.get("/get-bird-detections")
def get_bird_detections():
    if not db:
        raise HTTPException(status_code=500, detail="Firestore not initialized")

    try:
        docs = db.collection("bird_detections").stream()
        species_count = {}
        for doc in docs:
            data = doc.to_dict()
            species = data.get("class_name", "Unknown")
            count = data.get("count", 1)
            species_count[species] = species_count.get(species, 0) + count

        detections = [{"class_name": name, "count": count} for name, count in species_count.items()]
        detections.sort(key=lambda x: x["count"], reverse=True)
        return {"detections": detections}

    except Exception as e:
        print("Error fetching bird detections:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch detections")

@app.get("/get-bird-sightings")
def get_bird_sightings():
    return get_bird_detections()


BASE_URL = "http://127.0.0.1:8000"  # Replace with your deployed domain if not running locally

# --- Endpoints ---
@app.post("/process-image", response_model=DetectionResponse)
def process_image(req: DetectionRequest):
    try:
        image_data = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    height, width = image_np.shape[:2]
    timestamp = datetime.now().isoformat()
    detections = []

    try:
        results = model(image_np)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf < 0.2:
                    continue

                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Clamp coordinates within image bounds
                x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
                y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))

                species = None
                image_url = None

                if class_name == "bird" and gemini_model and (x2 > x1 and y2 > y1):
                    try:
                        # Improved cropping with padding but clamped within image bounds
                        padding = 10  # pixels of padding around the box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        bw = x2 - x1
                        bh = y2 - y1

                        # Make crop box a bit larger and square to better frame bird
                        crop_size = max(bw, bh) + 2 * padding
                        crop_x1 = max(0, cx - crop_size // 2)
                        crop_y1 = max(0, cy - crop_size // 2)
                        crop_x2 = min(width, cx + crop_size // 2)
                        crop_y2 = min(height, cy + crop_size // 2)

                        bird_crop = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
                        pil_img = Image.fromarray(bird_crop)

                        # Save cropped image
                        filename = f"bird_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        pil_img.save(filepath)
                        BASE_URL = "http://127.0.0.1:8000"  # Make sure this matches your backend URL
                        image_url = f"{BASE_URL}/static/{filename}"


                        # Identify species using Gemini
                        response = gemini_model.generate_content([
                            pil_img,
                            "What species of bird is shown in this image? Just return the species name."
                        ])
                        species = response.text.strip()
                        print(f"Gemini response: {species}")
                    except Exception as e:
                        print(f"Gemini error: {e}")

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    bird_species=species,
                    image_url=image_url
                ))

                # Save to Firestore
                if db and class_name == "bird" and species:
                    try:
                        db.collection("bird_detections").add({
                            "timestamp": timestamp,
                            "class_name": species,
                            "count": 1,
                            "image_url": image_url
                        })
                        print(f"Saved {species} to Firestore.")
                    except Exception as e:
                        print(f"Failed to write {species} to Firestore: {e}")

        return DetectionResponse(timestamp=timestamp, detections=detections)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")
@app.get("/get-recent-sightings")
def get_recent_sightings(limit: int = 10):
    if not db:
        raise HTTPException(status_code=500, detail="Firestore not initialized")
    try:
        docs = (
            db.collection("bird_detections")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        sightings = []
        for doc in docs:
            data = doc.to_dict()
            sightings.append({
                "species": data.get("class_name", "Unknown"),
                "timestamp": data.get("timestamp"),
                "image_url": data.get("image_url", "")
            })
        return sightings
    except Exception as e:
        print("Error fetching recent sightings:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch recent sightings")
