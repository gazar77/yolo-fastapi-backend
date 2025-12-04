from fastapi import FastAPI
import os, json, base64

app = FastAPI()
RESULTS_FILE = "static/results/results.json"

@app.get("/api/results/full")
async def get_full_results():
    if not os.path.exists(RESULTS_FILE):
        return {"error": "No results found"}

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    with open(data["annotated_image_url"].replace("/static/", "static/"), "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")

    data["annotated_image_base64"] = b64
    return data
