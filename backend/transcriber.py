#!/usr/bin/env python3

import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
import uvicorn
from common import emit_to_websocket

MODEL_NAME = "tiny.en"

PROMPT = """
Fire dispatch, SAR 2 enroute to Hippy Hole.
10-4
Fire dispatch, Medic 6.
Medic 6.
Have we made contact with the reporting party?
Negative, Medic 6. Still trying.
Copy.
(some time later)
"""

jobs = queue.Queue()
should_stop_transcriber_thread = False


class TranscriptionRequest(BaseModel):
    filename: str
    timestamp: str | None = None


def transcriber_worker():
    print("Loading Whisper model...")
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
    )

    print("Transcriber worker started.")

    while not should_stop_transcriber_thread:
        try:
            job = jobs.get(timeout=0.5)
        except queue.Empty:
            continue

        filename = job["filename"]

        try:
            segments, info = model.transcribe(
                filename,
                language="en",
                initial_prompt=PROMPT,
                beam_size=1,
                vad_filter=False,
            )

            text = "\n".join(
                s.text.strip()
                for s in segments
                if s.text.strip()
            )

            output = {
                "event_type": "SegmentEvent",
                "time": datetime.now().strftime("%F %T"),
                "text": text,
                "audio": filename
            }

            if len(text) == 0:
                if Path(filename).exists():
                    os.remove(filename)
                else:
                    print("Couldn't remove audio. File wasn't present.")
            else:
                out_file = Path(filename).with_suffix(".txt")

                with open(out_file, "w") as f:
                    json.dump(output, f)

                emit_to_websocket(output)

        except Exception as e:
            print(f"Failed to transcribe {filename}: {e}")

        finally:
            jobs.task_done()

    print("Transcriber worker exiting.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global should_stop_transcriber_thread

    threading.Thread(
        target=transcriber_worker,
        daemon=True,
    ).start()

    yield

    should_stop_transcriber_thread = True


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(req: TranscriptionRequest):
    if not os.path.exists(req.filename):
        raise HTTPException(
            status_code=404,
            detail="Audio file not found",
        )

    jobs.put(
        {
            "filename": req.filename,
            "timestamp": req.timestamp,
        }
    )

    return {
        "status": "queued",
        "filename": req.filename,
        "queue_depth": jobs.qsize(),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)