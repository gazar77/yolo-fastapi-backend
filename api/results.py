from fastapi import FastAPI
import os, json

app = FastAPI()

RESULTS_FILE = "static/results/results.json"

@app.get("/api/results")
async def get_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {"error": "No results found"}
