
import asyncio
from asyncio.subprocess import PIPE
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from pydantic import BaseModel
import threading
import time
import uvicorn


BASE_DIR = os.getcwd()


should_stop_camera_thread = False

# Global list to keep track of connected websocket clients
active_websocket_connections = set()

# ---------------------------------------------------------------------
# --------------------------- Payload models --------------------------
# ---------------------------------------------------------------------


class ArduinoCommand(BaseModel):
    command_string: str

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


@app.get("/")
async def root():
    filepath = os.path.join(BASE_DIR, "index.html")
    return FileResponse(filepath)


@app.post("/event")
async def receive_event(event_data: dict):
    await broadcast_to_websockets(event_data)
    return {"status": "ok"}


@app.get("/{path:path}")
async def serve_file(path: str):
    filepath = os.path.join(BASE_DIR, path)
    if os.path.isfile(filepath):
        return FileResponse(filepath)
    return {"error": "File not found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3011)
