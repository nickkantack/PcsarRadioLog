#!/usr/bin/env python3

import librosa
import random
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from kokoro import KPipeline


def change_speed(audio, speed, fs):
    # speed > 1.0 = faster, speed < 1.0 = slower
    return librosa.effects.time_stretch(
        audio,
        rate=speed
    )

TEXT = "Fire dispatch, SAR 2 is enroute to Hippy Hole."

VOICES = [
    "am_adam",
    "am_michael",
    "am_eric",
    "af_bella",
    "af_sarah",
]

# Pick two different voices
VOICE1, VOICE2 = random.sample(VOICES, 2)

FS = 24000


def apply_volume_envelope(audio, fs):
    n = len(audio)

    volume_changes_per_second = 2

    # Make sure we have at least two interpolation points
    points = max(2, int(n / fs * volume_changes_per_second))

    gains = np.random.uniform(0.1, 1.0, points)

    # Smooth interpolation across the clip
    envelope = np.interp(
        np.arange(n),
        np.linspace(0, n - 1, points),
        gains
    )

    return audio * envelope


def prepare_voice(pipeline, voice):
    """Generate and apply voice-specific processing."""

    _, _, audio = next(pipeline(TEXT, voice=voice))

    x = np.asarray(audio, dtype=np.float32)

    # Speed modulation - skipped now to avoid phaseing the two voices relative to one another
    """
    speed = np.random.uniform(1.2, 1.7)
    x = change_speed(x, speed, FS)

    # Volume modulation
    x = apply_volume_envelope(x, FS)
    """

    return x


def average_waveforms(a, b):
    """
    Make both waveforms the same length and literally average
    their samples together.
    """

    max_len = max(len(a), len(b))

    # Zero-pad the shorter waveform
    a_padded = np.pad(
        a,
        (0, max_len - len(a)),
        mode="constant"
    )

    b_padded = np.pad(
        b,
        (0, max_len - len(b)),
        mode="constant"
    )

    # Literal waveform averaging
    return (a_padded + b_padded) / 2.0


def align_starts(a, b, threshold=0.01):
    """Align the first significant audio in two waveforms."""
    def first_sound(x):
        idx = np.flatnonzero(np.abs(x) > threshold)
        return idx[0] if len(idx) else 0

    start_a = first_sound(a)
    start_b = first_sound(b)

    if start_a < start_b:
        a = np.pad(a, (start_b - start_a, 0))
    elif start_b < start_a:
        b = np.pad(b, (start_a - start_b, 0))

    # Make equal length
    n = max(len(a), len(b))
    a = np.pad(a, (0, n - len(a)))
    b = np.pad(b, (0, n - len(b)))

    return a, b


def dtw_align(x1, x2, fs):
    A = librosa.feature.mfcc(y=x1, sr=fs, n_mfcc=20)
    B = librosa.feature.mfcc(y=x2, sr=fs, n_mfcc=20)

    _, p = librosa.sequence.dtw(X=A, Y=B, metric="cosine")
    p = p[::-1]

    # x1-frame -> x2-frame mapping
    i, j = p[:, 0], p[:, 1]
    i, u = np.unique(i, return_index=True)
    j = j[u]

    # Convert frame mapping to sample mapping
    xi = librosa.frames_to_samples(i)
    xj = librosa.frames_to_samples(j)

    # For every x1 sample, find corresponding x2 sample
    x2pos = np.interp(
        np.arange(len(x1)),
        xi,
        xj
    )

    # Resample x2 along that warped timeline
    x2 = np.interp(
        x2pos,
        np.arange(len(x2)),
        x2
    )

    return x1, x2

# ------------------ TTS ------------------

pipeline = KPipeline(lang_code="a", device="cpu")

x1 = prepare_voice(pipeline, VOICE1)
x2 = prepare_voice(pipeline, VOICE2)

print(f"Selected voices: {VOICE1}, {VOICE2}")

# ------------------ Average the two voices ------------------
x1, x2 = align_starts(x1, x2)
x1, x2 = dtw_align(x1, x2, FS)
x = average_waveforms(x1, x2)

# ------------------ Radio effect ------------------

# Bandpass (telephone/radio)
sos = butter(
    6,
    [300, 2800],
    btype="bandpass",
    fs=FS,
    output="sos"
)

x = sosfilt(sos, x)

# Compression / saturation
x *= 8
x = np.tanh(x)

# Bit-depth reduction
levels = 48
x = np.round(x * levels) / levels

# Slight flutter
t = np.arange(len(x)) / FS
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

sf.write("radio.wav", x, FS)

print(
    f"Wrote radio.wav using voices "
    f"'{VOICE1}' + '{VOICE2}'"
)