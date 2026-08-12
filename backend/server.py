
import asyncio
from asyncio.subprocess import PIPE
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from pathlib import Path
from pydantic import BaseModel
import threading
import time
import uvicorn


BASE_DIR = os.getcwd()

segments_buffer = deque(maxlen=100)

should_stop_camera_thread = False

# Global list to keep track of connected websocket clients
active_websocket_connections = set()

# ---------------------------------------------------------------------
# --------------------------- Payload models --------------------------
# ---------------------------------------------------------------------

class SaveRequest(BaseModel):
    json: dict

# ---------------------------------------------------------------------
# --------------------------- Worker logic ----------------------------
# ---------------------------------------------------------------------

def camera_worker():
    print("Starting the camera worker")
    while not should_stop_camera_thread:
        time.sleep(1)

async def broadcast_to_websockets(data):
    """Broadcast data to all active websocket connections"""
    disconnected = set()
    for ws in active_websocket_connections:
        try:
            print("Sending websocket event")
            await ws.send_json(data)
        except Exception:
            # Keep track of disconnected clients to remove them
            disconnected.add(ws)
    
    # Remove disconnected clients
    for ws in disconnected:
        active_websocket_connections.discard(ws)


# We don't need a synchronous version since we're using the HTTP endpoint
# which properly handles the async broadcasting


@asynccontextmanager
async def lifespan(app: FastAPI):
    global should_stop_camera_thread, should_stop_arduino_thread
    # Do starting things before the "yield"
    threading.Thread(target=camera_worker, daemon=True).start()
    yield
    # Do teardown things after the "yield"
    should_stop_camera_thread = True

# ---------------------------------------------------------------------
# --------------------------- App configuration -----------------------
# ---------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)
# Mount static files for serving index.html and assets
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.websocket("/transcription_stream")
async def stats_stream(ws: WebSocket):
    await ws.accept()
    active_websocket_connections.add(ws)
    
    try:
        while True:
            # Keep the connection alive
            await asyncio.sleep(1)
    except Exception:
        pass
    finally:
        active_websocket_connections.discard(ws)
        await ws.close()




@app.get("/event_buffer")
async def event_buffer():
    return list(segments_buffer)


@app.get("/")
async def root():
    filepath = os.path.join(BASE_DIR, "index.html")
    return FileResponse(filepath)


@app.post("/event")
async def receive_event(event_data: dict):
    if event_data["event_type"] == "SegmentEvent":
        segments_buffer.appendleft(event_data)
    await broadcast_to_websockets(event_data)
    return {"status": "ok"}


@app.get("/label/files")
def label_files():
    ids = sorted(p.stem for p in Path("data").glob("*.wav"))
    ids_to_vend = []
    for id in ids:
        if Path(f"data/{id}.txt").exists():
            with open(f"data/{id}.txt", "r") as file:
                ob = json.loads(file.read())
                if "label" not in ob:
                    ids_to_vend.append(id)
    return ids_to_vend


@app.get("/label/item/{item_id}")
def label_item(item_id: str):
    txt = Path(f"data/{item_id}.txt")
    wav = Path(f"data/{item_id}.wav")

    if not txt.exists() or not wav.exists():
        raise HTTPException(status_code=404)

    with txt.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "id": item_id,
        "json": payload,
        "audio": f"/data/{wav.name}",
    }


@app.post("/label/item/{item_id}")
def save_label(item_id: str, req: SaveRequest):
    txt = Path(f"data/{item_id}.txt")

    if not txt.exists():
        raise HTTPException(status_code=404)

    with txt.open("w", encoding="utf-8") as f:
        json.dump(req.json, f, indent=2, ensure_ascii=False)

    return {"ok": True}


@app.get("/{path:path}")
async def serve_file(path: str):
    filepath = os.path.join(BASE_DIR, path)
    if os.path.isfile(filepath):
        return FileResponse(filepath)
    return {"error": "File not found"}


if __name__ == "__main__":

    # TODO attempt to hydrate the segment_buffer with
    # segments from the disk
    now = int(time.time()*1000000)
    audio_files = os.listdir("data")
    times = list(map(lambda x: x.split("_")[1], list(filter(lambda x: x[-4:] == ".wav", audio_files))))
    print(times)
    cursor = 1
    while cursor <= 100 and cursor < len(times) - 1 and float(times[-cursor]) > now - 30 * 60 * 1E6:
        cursor += 1
    while cursor > 0:
        with open(f"data/{audio_files[cursor].replace('.wav', '.txt')}", "r") as file:
            segment = json.loads(file.read())
            segments_buffer.append(segment)
        cursor -= 1
    print(segments_buffer)

    uvicorn.run(app, host="0.0.0.0", port=3011)
