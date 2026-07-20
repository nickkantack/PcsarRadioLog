#!/usr/bin/env python3

import librosa
import random
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d
from kokoro import KPipeline


def change_speed(audio, speed, fs):
    # speed > 1.0 = faster, speed < 1.0 = slower
    return librosa.effects.time_stretch(
        audio,
        rate=speed
    )

TEXT = "Fire dispatch, SAR 2 is enroute to Hippy Hole."

VOICE = random.choice([
    "am_adam",
    "am_michael",
    "am_eric",
    "af_bella",
    "af_sarah",
])

def apply_volume_envelope(audio, fs):
    n = len(audio)

    volume_changes_per_second = 2

    # Random gain points every ~1 second
    points = n // fs * volume_changes_per_second
    gains = np.random.uniform(0.1, 1.0, points)

    # Smooth interpolation across the clip
    envelope = np.interp(
        np.arange(n),
        np.linspace(0, n, points),
        gains
    )

    plt.plot(envelope)
    plt.show()

    return audio * envelope

# ------------------ TTS ------------------

pipeline = KPipeline(lang_code="a")
_, _, audio = next(pipeline(TEXT, voice=VOICE))

x = np.asarray(audio, dtype=np.float32)
fs = 24000

# Speed modulation
speed = np.random.uniform(1.2, 1.7)
x = change_speed(x, speed, fs)

# Volume modulation
x = apply_volume_envelope(x, fs)

# ------------------ Radio effect ------------------

# Bandpass (telephone/radio)
sos = butter(6, [300, 2800], btype="bandpass", fs=fs, output="sos")
x = sosfilt(sos, x)

# Compression / saturation
x *= 8
x = np.tanh(x)

# Bit-depth reduction
levels = 48
x = np.round(x * levels) / levels

# Slight flutter
t = np.arange(len(x)) / fs
x *= 0.95 + 0.05 * np.sin(2 * np.pi * 30 * t)

# Hiss
x += np.random.normal(0, 0.015, len(x))

# Random crackles
mask = np.random.random(len(x)) < 7e-4
x[mask] += np.random.uniform(-0.8, 0.8, mask.sum())

# Final limiting
x *= 1.8
x = np.clip(x, -1, 1)

# ------------------ Output ------------------

sf.write("radio.wav", x, fs)
print(f"Wrote radio.wav using voice '{VOICE}'")
