#!/usr/bin/env python3

import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import pickle
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from common import emit_to_websocket
import torch
import torchaudio
from denoise_prototype import Denoiser
from constants import SAMPLE_RATE
from constrained_generation import TrieLogitBias, UnigramLogitBias

MODEL_NAME = "small.en"

MODEL_TO_LOAD = "alpha"

PROMPT = """"""

jobs = queue.Queue()
should_stop_transcriber_thread = False

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

DEVICE = "cpu"

class TranscriptionRequest(BaseModel):
    filename: str
    timestamp: str | None = None


def transcriber_worker():
    print("Loading Whisper model...")

    base_model = "openai/whisper-small.en"
    processor = WhisperProcessor.from_pretrained(
        base_model,
        language="english",
        task="transcribe"
    )
    whisper = WhisperForConditionalGeneration.from_pretrained(base_model).to(DEVICE)
    whisper.eval()  # Keep whisper in eval mode since we're not training it

    denoiser = Denoiser().to(DEVICE)
    denoiser.add_input = True
    denoiser.eval()
    saved_object = torch.load(f"released_models/{MODEL_TO_LOAD}/{MODEL_TO_LOAD}_denoiser.pt")
    denoiser.load_state_dict(saved_object["model"])

    ngram_biases = None
    unigram_biases = None
    top_phrases = None
    with open(f"released_models/{MODEL_TO_LOAD}/{MODEL_TO_LOAD}_ngram_biases_30", "rb") as file:
        ngram_biases = pickle.load(file)
    with open(f"released_models/{MODEL_TO_LOAD}/{MODEL_TO_LOAD}_unigram_biases_30", "rb") as file:
        unigram_biases = pickle.load(file)
    # TODO load top phrases too

    logits_processors = []
    """
    logits_processor = [
        TrieLogitBias(trie),
        UnigramLogitBias(unigram_biases)
    ]
    """

    print("Transcriber worker started.")

    with torch.no_grad():
        while not should_stop_transcriber_thread:
            try:
                job = jobs.get(timeout=0.5)
            except queue.Empty:
                continue

            filename = job["filename"]

            try:

                waveform, sr = torchaudio.load(filename)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

                waveform = waveform.mean(0)

                mel = processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                ).input_features.to(DEVICE)

                if denoiser:
                    mel = denoiser(mel)

                generated_ids = whisper.generate(
                        input_features=mel,
                        num_beams=5,
                        do_sample=False,
                        early_stopping=True,
                        logits_processor=logits_processors
                    )

                text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
                
                output = {
                    "event_type": "SegmentEvent",
                    "time": datetime.now().strftime("%F %T"),
                    "text": text,
                    "audio": filename
                }

                if len(text) == 0 or re.fullmatch(r"[^a-zA-Z]*", text):
                    # Empty text and text with no alpha characters (e.g. "...") shouldn't be further processed
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