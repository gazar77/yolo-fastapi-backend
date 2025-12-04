from fastapi import FastAPI
import os, json

app = FastAPI()
HISTORY_FILE = "static/results/history.json"

@app.delete("/api/history/{item_id}")
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
