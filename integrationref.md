# 🚀 Traffic Violation Detection System — 1 Day Build Plan

## 🎯 Goal

Build a working system that:

* Accepts **image/video upload**
* Detects **traffic violations**
* Displays **annotated output + violation labels**

---

# ⏱️ DAY PLAN (STRICT)

| Time      | Task                  |
| --------- | --------------------- |
| 0–1 hr   | Setup environment     |
| 1–3 hr   | YOLO detection setup  |
| 3–5 hr   | Violation logic       |
| 5–7 hr   | FastAPI backend       |
| 7–10 hr  | Frontend UI           |
| 10–12 hr | Integration + testing |

---

# 🧱 STEP 1: PROJECT SETUP

## Create structure

```
traffic-ai/
│
├── backend/
│   ├── main.py
│   ├── detector.py
│   ├── violations.py
│   └── utils.py
│
├── frontend/
│   └── (Lovable export or Next.js app)
│
├── models/
│   └── yolov8n.pt
│
└── requirements.txt
```

---

## Install dependencies

```bash
pip install ultralytics opencv-python fastapi uvicorn python-multipart
```

---

# 🧠 STEP 2: YOLO DETECTION

## detector.py

```python
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

def detect_objects(frame):
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class": cls,
                "confidence": conf,
                "box": xyxy
            })

    return detections
```

---

# 🚨 STEP 3: VIOLATION LOGIC

## violations.py

```python
def detect_violations(detections):
    violations = []

    for d in detections:
        cls = d["class"]

        # Example mapping (simplified)
        if cls == 0:  # person
            violations.append("Possible pedestrian violation")

        if cls == 1:  # bicycle
            violations.append("Wrong lane possible")

        if cls == 2:  # car
            pass

    return list(set(violations))
```

👉 Real systems combine detection + tracking + spatial rules to identify violations ([ijcaonline.org](https://ijcaonline.org/archives/volume186/number79/kathait-2025-ijca-924714.pdf?utm_source=chatgpt.com "Deep Learning-based Approach for Detecting Traffic ..."))

---

# 🎥 STEP 4: VIDEO PROCESSING

## utils.py

```python
import cv2
from detector import detect_objects
from violations import detect_violations

def process_video(path):
    cap = cv2.VideoCapture(path)
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)
        violations = detect_violations(detections)

        results.append({
            "violations": violations
        })

    cap.release()
    return results
```

👉 Videos are processed frame-by-frame and analyzed continuously ([rjwave.org](https://rjwave.org/ijedr/papers/IJEDR2601789.pdf?utm_source=chatgpt.com "Traffix: An Automated Traffic Violation Detection System ..."))

---

# ⚙️ STEP 5: FASTAPI BACKEND

## main.py

```python
from fastapi import FastAPI, UploadFile, File
import shutil
from utils import process_video
from detector import detect_objects
import cv2

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".mp4"):
        result = process_video(file_path)
    else:
        img = cv2.imread(file_path)
        detections = detect_objects(img)
        result = {"violations": detections}

    return {"result": result}
```

---

## Run server

```bash
uvicorn main:app --reload
```

---

# 🎨 STEP 6: FRONTEND (LOVABLE / NEXT.JS)

## Requirements for Codex

* Upload button
* Preview image/video
* Call API `/upload`
* Show:
  * Violations list
  * Annotated media (optional)

---

## Example API call

```javascript
const formData = new FormData();
formData.append("file", file);

const res = await fetch("http://localhost:8000/upload", {
  method: "POST",
  body: formData
});

const data = await res.json();
```

---

# 🧪 STEP 7: TESTING

Test with:

* Image (helmet / no helmet)
* Traffic video
* Edge cases:
  * No detections
  * Invalid file

---

# 🔥 STEP 8: IMPROVE (IF TIME LEFT)

Add:

* Bounding boxes on output
* Confidence scores
* Better violation mapping:
  * Helmet → head detection
  * Red light → line crossing logic
  * Phone usage → object near face

---

# 🧠 HOW SYSTEM WORKS (IMPORTANT FOR VIVA)

Pipeline:

1. Upload media
2. Frame extraction (video)
3. YOLO detection
4. Rule-based violation logic
5. Output visualization

This follows standard CV pipeline:
Detection → Tracking → Logic → Output ([Medium](https://medium.com/%40haris2bashir9/building-a-production-ready-traffic-violation-detection-system-with-computer-vision-a9839fdd260c?utm_source=chatgpt.com "Building a Production-Ready Traffic Violation Detection ..."))

---

# ⚠️ LIMITATIONS (Say this if asked)

* No custom trained model
* Basic violation logic
* No tracking (DeepSORT not added)

---

# ✅ FINAL OUTPUT EXPECTED

* Upload → Detect → Show violations
* Works for both image & video
* Clean UI (Lovable)
* Fast response

---

# 🧨 CODEx FINAL INSTRUCTION

Give Codex this:

"Complete backend integration, improve violation logic, and connect frontend UI to API. Add bounding box visualization."

---
