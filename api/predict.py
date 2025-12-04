from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io, uuid, json, os, datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = YOLO("best.pt")
os.makedirs("static/results", exist_ok=True)

RESULTS_FILE = "static/results/results.json"
HISTORY_FILE = "static/results/history.json"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    results = model(img_np)
    detections = json.loads(results[0].to_json())

    unique_name = uuid.uuid4().hex
    annotated_name = f"{unique_name}_annotated.jpg"
    annotated_path = f"static/results/{annotated_name}"

    annotated_img = results[0].plot()
    Image.fromarray(annotated_img).save(annotated_path)

    output_data = {
        "annotated_image_url": f"/static/results/{annotated_name}",
        "detections": detections
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

    history_item = {
        "id": unique_name,
        "date": str(datetime.datetime.now()),
        "annotated_image_url": f"/static/results/{annotated_name}",
        "detections": detections
    }

    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)

    history_data.append(history_item)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=4)

    return output_data
