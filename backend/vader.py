#!/usr/bin/env python3
import time
import uuid
from datetime import datetime

import requests
import soundfile as sf

from audio_capture import AudioCapture
from vad import VadState, get_vad_state
from common import emit_to_transcriber, emit_to_server


SAMPLE_RATE = 16000

AUDIO_BUFFER_MS = 20000
DECISION_INTERVAL_MS = 1200

TRANSCRIBER_URL = "http://transcriber:8000/transcribe"

print("Running")

# TODO dynamically set sample rate in line below to one 
# supported by the chosen device. This is required.
# transcriber.py will resample to 16000 regardless of
# the choice here. We just need this to make the underlying
# device happy so this script doesn't crash.
audio = AudioCapture(AUDIO_BUFFER_MS, sample_rate=44100)
audio.resume()

is_speaking = False
speech_start = time.monotonic()
no_vad_until = 0
is_speaking_unlocks_at = 0

try:
    while True:
        now = time.monotonic()

        if now < no_vad_until:
            time.sleep(0.1)
            continue

        no_vad_until = now + DECISION_INTERVAL_MS / 1000 / 2

        recent = audio.get(DECISION_INTERVAL_MS)

        vad_state = get_vad_state(
            recent,
            SAMPLE_RATE,
            250,
            is_speaking,
            False
        )
        # vad_state = VadState.SpeechPresent if has_speech(recent, SAMPLE_RATE) else VadState.SpeechAbsent

        do_save = False
        was_cut_off = False
        decision_interval_is_silent = False

        if not is_speaking and vad_state == VadState.SpeechPresent:
            speech_start = now - DECISION_INTERVAL_MS / 1000 / 2
            is_speaking = True
            is_speaking_unlocks_at = now + 1
            emit_to_server({
                "event_type": "SpeechStartEvent",
                "time": datetime.now().strftime("%F %T")
            })
            continue

        if is_speaking and vad_state == VadState.SpeechAbsent and now > is_speaking_unlocks_at:
            is_speaking = False
            do_save = True
            decision_interval_is_silent = True
            emit_to_server({
                "event_type": "SpeechStopEvent",
                "time": datetime.now().strftime("%F %T")
            })
            # We got a clean speech stop. Let's transcribe what we got
            # and then wait long enough that we only transcribe new
            # audio.
            no_vad_until = now + DECISION_INTERVAL_MS / 1000

        elif is_speaking:
            # Protect the end of the speech
            if (now - speech_start) * 1000 > AUDIO_BUFFER_MS - DECISION_INTERVAL_MS:
                do_save = True
                was_cut_off = True

        if not do_save:
            continue

        duration_ms = int((now - speech_start) * 1000)
        pcm = audio.get(min(duration_ms, AUDIO_BUFFER_MS))

        if decision_interval_is_silent:
            # Chop off the last 80% of the last DECISION_INTERVAL_MS of audio
            # because the last DECISION_INTERVAL_MS of audio is presumed blank.
            pcm = pcm[:int(-SAMPLE_RATE * DECISION_INTERVAL_MS / 1000 * 0.8)]

        ident = f"data/segment_{int(time.time()*1000000)}_{uuid.uuid4()}"
        wav_filename = ident + ".wav"

        sf.write(
            wav_filename,
            pcm,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        event_time = datetime.now().strftime("%F %T")

        emit_to_transcriber({
            "event_type": "TranscriptionQueuedEvent",
            "time": datetime.now().strftime("%F %T"),
            "filename": wav_filename,
            "continuing": was_cut_off
        })

        speech_start = time.monotonic()

except KeyboardInterrupt:
    pass

finally:
    audio.pause()
