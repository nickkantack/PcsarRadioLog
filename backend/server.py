
import asyncio
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


@app.websocket("/stats_stream")
async def stats_stream(ws: WebSocket):
    global stat_index
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps({
                "time": time.time()    
            }))
            await asyncio.sleep(1)
    except Exception:
        pass  # client disconnected
    finally:
        await ws.close()


@app.get("/")
async def root():
    filepath = os.path.join(BASE_DIR, "index.html")
    return FileResponse(filepath)


@app.get("/{path:path}")
async def serve_file(path: str):
    filepath = os.path.join(BASE_DIR, path)
    if os.path.isfile(filepath):
        return FileResponse(filepath)
    return {"error": "File not found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3011)
