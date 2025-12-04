from fastapi import FastAPI
from fastapi.responses import FileResponse
import os, json

app = FastAPI()
RESULTS_FILE = "static/results/results.json"

@app.get("/api/results/image")
async def get_annotated_image():
    if not os.path.exists(RESULTS_FILE):
        return {"error": "No results found"}

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    img_path = data["annotated_image_url"].replace("/static/", "static/")
    if not os.path.exists(img_path):
        return {"error": "Image not found"}

    return FileResponse(img_path, media_type="image/jpeg")
