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

app = FastAPI()

# ======================
# تمكين CORS للفرونت
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
except Exception as e:
    print("❌ Error loading model:", e)

# ======================
# مسارات التخزين
# ======================
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

RESULTS_FILE = "static/results/results.json"
HISTORY_FILE = "static/results/history.json"

# إنشاء ملف History لو مش موجود
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ======================
# 🔥 API: استقبال الصورة وتشغيل الموديل
# ======================
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # قراءة الصورة
        img_bytes = await image.read()

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        img_np = np.array(img)

        # تشغيل الموديل
        results = model(img_np)

        # قراءة الديتكشن JSON
        try:
            detections = json.loads(results[0].to_json())
        except:
            detections = []

        # اسم فريد
        unique_name = uuid.uuid4().hex
        annotated_name = f"{unique_name}_annotated.jpg"
        annotated_path = f"static/results/{annotated_name}"

        # 🔥 حفظ الصورة المعلمة
        try:
            annotated_img = results[0].plot()   # numpy
            annotated_pil = Image.fromarray(annotated_img)
            annotated_pil.save(annotated_path)
        except Exception as e:
            print("❌ Error saving annotated image:", e)
            return JSONResponse({"error": "Failed to save annotated image"}, status_code=500)

        # حفظ JSON للنتيجة الأخيرة
        output_data = {
            "annotated_image_url": f"/static/results/{annotated_name}",
            "annotated_image_path": annotated_path,
            "detections": detections
        }

        with open(RESULTS_FILE, "w") as f:
            json.dump(output_data, f, indent=4)

        # ======================
        # إضافة التحليل للـ History
        # ======================
        history_item = {
            "id": unique_name,
            "date": str(datetime.datetime.now()),
            "annotated_image_url": f"/static/results/{annotated_name}",
            "detections": detections
        }

        # قراءة التاريخ القديم
        with open(HISTORY_FILE, "r") as f:
            history_data = json.load(f)

        history_data.append(history_item)

        # حفظ التاريخ مرة تانية
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_data, f, indent=4)

        return output_data

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ======================
# GET: يرجع JSON فقط للنتيجة الأخيرة
# ======================
@app.get("/results")
async def get_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {"error": "No results found"}

# ======================
# GET: JSON + Base64 Image
# ======================
@app.get("/results/full")
async def get_full_results():
    if not os.path.exists(RESULTS_FILE):
        return {"error": "No results found"}

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    img_path = data["annotated_image_path"]

    with open(img_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")

    data["annotated_image_base64"] = b64
    return data

# ======================
# GET: يرجع الصورة نفسها
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
# GET: History (كل التحاليل السابقة)
# ======================
@app.get("/history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# ======================
# DELETE: مسح كل الـ History
# ======================
# ======================
# DELETE: مسح عنصر واحد من History
# ======================
@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    if not os.path.exists(HISTORY_FILE):
        return {"error": "History file not found"}

    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)

    # البحث عن العنصر
    new_history = [item for item in history_data if item["id"] != item_id]

    # إذا لم نجد العنصر
    if len(new_history) == len(history_data):
        return {"error": "Item not found"}

    # حفظ التاريخ الجديد بدون العنصر
    with open(HISTORY_FILE, "w") as f:
        json.dump(new_history, f, indent=4)

    # حذف الصورة المرتبطة بالعنصر
    for item in history_data:
        if item["id"] == item_id:
            img_path = os.path.join("static/results", os.path.basename(item["annotated_image_url"]))
            if os.path.exists(img_path):
                os.remove(img_path)
            break

    return {"message": f"Item {item_id} deleted successfully"}


# ======================
# تشغيل السيرفر
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
