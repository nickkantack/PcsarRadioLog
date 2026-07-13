#!/usr/bin/env python3

import json
import time
import uuid
from datetime import datetime

import requests
import soundfile as sf

from audio_capture import AudioCapture
from vad import VadState, get_vad_state


SAMPLE_RATE = 16000

AUDIO_BUFFER_MS = 20000
DECISION_INTERVAL_MS = 2000

TRANSCRIBER_URL = "http://127.0.0.1:8000/transcribe"


def emit(event_type, **fields):
    event = {
        "type": event_type,
        "time": datetime.now().strftime("%F %T"),
    }
    event.update(fields)
    print(json.dumps(event), flush=True)


print("Running")

audio = AudioCapture(AUDIO_BUFFER_MS)
audio.resume()

emit("InitializationEvent")

is_speaking = False
speech_start = time.monotonic()
last_inference = time.monotonic()

try:
    while True:
        now = time.monotonic()

        if (now - last_inference) * 1000 < DECISION_INTERVAL_MS / 2:
            time.sleep(0.1)
            continue

        last_inference = now

        recent = audio.get(DECISION_INTERVAL_MS)

        vad_state = get_vad_state(
            recent,
            SAMPLE_RATE,
            250,
            is_speaking,
            False
        )

        do_save = False
        was_cut_off = False

        if not is_speaking and vad_state == VadState.SpeechPresent:
            speech_start = now - DECISION_INTERVAL_MS / 1000
            is_speaking = True
            emit("SpeechStartEvent")
            continue

        if is_speaking and vad_state == VadState.SpeechAbsent:
            is_speaking = False
            do_save = True
            emit("SpeechStopEvent")

        elif (
            is_speaking
            and (now - speech_start) * 1000
            > AUDIO_BUFFER_MS - DECISION_INTERVAL_MS
        ):
            do_save = True
            was_cut_off = True

        if not do_save:
            continue

        duration_ms = int((now - speech_start) * 1000)
        pcm = audio.get(min(duration_ms, AUDIO_BUFFER_MS))

        ident = f"data/segment_{int(time.time()*1000000)}_{uuid.uuid4()}"
        wav_filename = ident + ".wav"

        sf.write(
            wav_filename,
            pcm,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        event_time = datetime.now().strftime("%F %T")

        emit(
            "TranscriptionQueuedEvent",
            filename=wav_filename,
            continuing=was_cut_off,
        )

        try:
            requests.post(
                TRANSCRIBER_URL,
                json={
                    "filename": wav_filename,
                    "timestamp": event_time,
                },
                timeout=0.5,
            )
        except Exception as e:
            emit(
                "TranscriptionQueueFailedEvent",
                filename=wav_filename,
                error=str(e),
            )

        speech_start = time.monotonic()

except KeyboardInterrupt:
    pass

finally:
    audio.pause()