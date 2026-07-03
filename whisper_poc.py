
import collections
import numpy as np
import sounddevice as sd
from pywhispercpp.model import Model

# Configuration parameters
SAMPLE_RATE = 16000     # Whisper strictly requires 16kHz
STEP_DURATION = 1.0     # Frequency of updates (how often it checks for new audio)
WINDOW_DURATION = 5.0   # Context size passed to Whisper (5 seconds)
CHANNELS = 1            # Mono input

# Initialize model
model = Model('tiny.en', n_threads=4)

# Calculate sample counts
step_samples = int(STEP_DURATION * SAMPLE_RATE)
window_samples = int(WINDOW_DURATION * SAMPLE_RATE)

# Create a fixed-size ring buffer using collections.deque
# maxlen ensures that when new data enters, old data drops off automatically
audio_buffer = collections.deque(maxlen=window_samples)
# Pre-fill buffer with silence so it starts completely full
audio_buffer.extend(np.zeros(window_samples, dtype=np.float32))

print("🎤 Listening with sliding window... Press Ctrl+C to stop.")

try:
    while True:
        # 1. Record the short 'step' increment of fresh audio
        fresh_chunk = sd.rec(
            step_samples, 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype='float32'
        )
        sd.wait()

        # 2. Push the fresh audio into our sliding ring buffer
        audio_buffer.extend(fresh_chunk.flatten())

        # 3. Convert the buffer contents back into a flat NumPy array for Whisper
        current_window = np.array(audio_buffer, dtype=np.float32)

        # 4. Transcribe the whole context window
        segments = model.transcribe(current_window)
        
        # 5. Extract and clean the last segment to avoid repeating old text
        text_segments = [seg.text.strip() for seg in segments if seg.text.strip()]
        if text_segments:
            # Print only the latest output detected in the context window
            print(f"💬 {text_segments[-1]}", flush=True)

except KeyboardInterrupt:
    print("\n🛑 Stopped listening.")
