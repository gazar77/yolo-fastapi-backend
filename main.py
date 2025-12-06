import os
import io
import json
import base64
import uuid
import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ======================
# إنشاء التطبيق
# ======================
app = FastAPI()

# ======================
# تفعيل CORS
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# تحميل الموديل
# ======================
try:
    model = YOLO("best.pt")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# ======================
# تجهيز المسارات
# ======================
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

RESULTS_FILE = "static/results/results.json"
HISTORY_FILE = "static/results/history.json"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ======================
# ✅ POST /predict
# ======================
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        img_np = np.array(img)

        results = model(img_np)

        try:
            detections = json.loads(results[0].to_json())
        except:
            detections = []

        unique_name = uuid.uuid4().hex
        annotated_name = f"{unique_name}_annotated.jpg"
        annotated_path = f"static/results/{annotated_name}"

        annotated_img = results[0].plot()
        annotated_pil = Image.fromarray(annotated_img)
        annotated_pil.save(annotated_path)

        output_data = {
            "annotated_image_url": f"/static/results/{annotated_name}",
            "annotated_image_path": annotated_path,
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

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ======================
# ✅ GET آخر نتيجة
# ======================
@app.get("/results")
async def get_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {"error": "No results found"}

# ======================
# ✅ GET نتيجة + Base64
# ======================
@app.get("/results/full")
async def get_full_results():
    if not os.path.exists(RESULTS_FILE):
        return {"error": "No results found"}

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    with open(data["annotated_image_path"], "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")

    data["annotated_image_base64"] = b64
    return data

# ======================
# ✅ GET الصورة فقط
# ======================
@app.get("/results/image")
async def get_annotated_image():
    if not os.path.exists(RESULTS_FILE):
        return {"error": "No results found"}

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    img_path = data["annotated_image_path"]

    if not os.path.exists(img_path):
        return {"error": "Image not found"}

    return FileResponse(img_path, media_type="image/jpeg")

# ======================
# ✅ GET كل history
# ======================
@app.get("/history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# ======================
# ✅ DELETE عنصر واحد من history
# ======================
@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    if not os.path.exists(HISTORY_FILE):
        return {"error": "History file not found"}

    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)

    new_history = [item for item in history_data if item["id"] != item_id]

    if len(new_history) == len(history_data):
        return {"error": "Item not found"}

    with open(HISTORY_FILE, "w") as f:
        json.dump(new_history, f, indent=4)

    for item in history_data:
        if item["id"] == item_id:
            img_path = os.path.join("static/results", os.path.basename(item["annotated_image_url"]))
            if os.path.exists(img_path):
                os.remove(img_path)
            break

    return {"message": "Deleted successfully"}

# ======================
# ✅ تشغيل Railway بطريقة صحيحة
# ======================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
