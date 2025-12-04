from fastapi import FastAPI
import os, json

app = FastAPI()
HISTORY_FILE = "static/results/history.json"

@app.get("/api/history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []
