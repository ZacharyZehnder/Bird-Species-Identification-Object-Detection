# Bird Species Detection and Tracking  
**Raspberry Pi 5 | AI Camera | MobileNetSSD | Gemini API | Firebase**  

---

## Abstract  
This project demonstrates the design and implementation of a **real-time bird species detection and classification system** utilizing a **Raspberry Pi 5** with an **AI Camera Module**.  
Captured video frames are processed by the **MobileNetSSD** model for bird detection. Confirmed detections are sent to the **Google Gemini API** for species classification. Results (species, timestamp, image) are stored in **Firebase Firestore** and displayed through a **web-based dashboard**.  

The system achieved an **average latency of ~250 milliseconds** and an overall **species classification accuracy of 93%**.  
**Development required 68 hours over an 11-week period.**  

---

## Introduction  
Automated bird detection and classification has applications in ecology, conservation, and environmental monitoring.  
This project was undertaken as an independent summer research initiative following competitive program rejections, with the objective of demonstrating applied skills in **computer vision, machine learning, and cloud integration**.  

The system serves as both a functional prototype for real-world bird monitoring and as a precursor to a planned senior thesis involving safety-focused machine learning applications.  

---

## Methods  

### Hardware  
- Raspberry Pi 5  
- Raspberry Pi AI Camera Module  
- Tripod mount and power supply  

### Software and Libraries  
- Python 3  
- OpenCV (image capture and processing)  
- MobileNetSSD (object detection)  
- FastAPI (backend API framework)  
- Firebase Admin SDK (database integration)  
- Google Gemini API (species classification)  
- Uvicorn (ASGI server)  

### Cloud and Storage  
- Firebase Firestore (data logging and image storage)  

### Frontend  
- HTML and Tailwind CSS (dashboard)  

---

### System Workflow  

1. **Video Capture** – Raspberry Pi AI Camera streams live video.  
2. **Object Detection** – MobileNetSSD model detects bird presence.  
3. **Species Identification** – Gemini API classifies species.  
4. **Data Logging** – Results stored in Firebase Firestore (species, timestamp, image).  
5. **Visualization** – UI dashboard displays recent and historical detections.  

---

## Results  

- **Latency:** Average of ~250ms per frame (near real-time).  
- **Accuracy:** 93% classification success rate, with minimal misclassifications.  
- **Visualization:** UI successfully displayed both real-time detections and history; occasional storage latency caused missing images.  

---

## Discussion  

This system demonstrates the feasibility of integrating lightweight deep learning models with cloud-based AI APIs on edge devices.  
The combination of **on-device detection (MobileNetSSD)** and **cloud classification (Gemini API)** balances computational efficiency with accuracy.  

### Future Work  
- Retraining MobileNetSSD with custom images to improve bird-specific detection.  
- Adding a **live video feed** to the dashboard interface.  
- Enabling **global access** to the UI rather than local-only hosting.  

---

## Installation & Setup  

### 1. Clone the Repository  
```bash
git clone https://github.com/<your-username>/bird-species-detection.git
cd bird-species-detection
