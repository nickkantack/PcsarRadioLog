#!/usr/bin/env python3

import json
import time
import uuid
from datetime import datetime

import soundfile as sf
from faster_whisper import WhisperModel

# Your existing implementations.
# audio.get(ms) must return the most recent ms of float32 mono audio at 16 kHz.
from backend.audio_capture import AudioCapture

# Your existing VAD implementation.
from vad import get_vad_state, VadState


SAMPLE_RATE = 16000
MODEL_NAME = "tiny.en"

AUDIO_BUFFER_MS = 20000
DECISION_INTERVAL_MS = 2000

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


def emit(event_type, **fields):
    event = {
        "type": event_type,
        "time": datetime.now().strftime("%F %T"),
    }
    event.update(fields)
    print(json.dumps(event), flush=True)

    # TODO have a client send the event over network is possible

    # TODO emit transcription requests and have a separate python
    # worker find the file, load the audio, run the transcription,
    # and write the resulting transcription. This file should be
    # responsible just for dropping audio files with logical
    # separation of speech.


print("Get the model")
model = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8",
)

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

        # Look at the tail of the ring buffer to determine if speech started/stopped.
        recent = audio.get(DECISION_INTERVAL_MS)

        vad_state = get_vad_state(
            recent,
            SAMPLE_RATE,
            250,
            False,
        )

        do_inference = False
        was_cut_off = False

        if not is_speaking and vad_state == VadState.SpeechPresent:
            speech_start = now - DECISION_INTERVAL_MS / 1000
            is_speaking = True
            emit("SpeechStartEvent")
            continue

        if is_speaking and vad_state == VadState.SpeechAbsent:
            is_speaking = False
            do_inference = True
            emit("SpeechStopEvent")

        elif is_speaking and (now - speech_start) * 1000 > AUDIO_BUFFER_MS - DECISION_INTERVAL_MS:
            do_inference = True
            was_cut_off = True

        if not do_inference:
            continue

        duration_ms = int((now - speech_start) * 1000)
        pcm = audio.get(min(duration_ms, AUDIO_BUFFER_MS))

        emit("InterferenceStartEvent")

        segments, info = model.transcribe(
            pcm,
            language="en",
            initial_prompt=PROMPT,
            beam_size=1,
            vad_filter=False,
        )

        emit("InterferenceStopEvent")

        found_segment = False

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            found_segment = True

            start = int(segment.start * SAMPLE_RATE)
            end = int(segment.end * SAMPLE_RATE)

            # Clamp in case model timestamps run slightly past the buffer.
            start = max(0, min(start, len(pcm)))
            end = max(0, min(end, len(pcm)))

            segment_pcm = pcm[start:end]

            if len(segment_pcm) == 0:
                continue

            suffix = " (continuing...)" if was_cut_off else ""

            output = {
                "type": "SegmentEvent",
                "time": datetime.now().strftime("%F %T"),
                "text": text + suffix,
            }

            ident = f"data/segment_{int(time.time()*1000000)}_{uuid.uuid4()}"

            sf.write(
                ident + ".wav",
                segment_pcm,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            with open(ident + ".txt", "w") as f:
                f.write(json.dumps(output))

            print(json.dumps(output), flush=True)

        if not found_segment:
            # print("No segments", flush=True)
            is_speaking = False

        # Preserve the same behavior as the C++ version:
        # if another utterance starts while inference is happening,
        # the next loop begins from here.
        speech_start = time.monotonic()


except KeyboardInterrupt:
    pass

finally:
    audio.pause()