import threading
import numpy as np
import sounddevice as sd


class AudioCapture:
    def __init__(self, length_ms, sample_rate=16000):
        self.sample_rate = sample_rate
        self.size = int(sample_rate * length_ms / 1000)

        self.buffer = np.zeros(self.size, dtype=np.float32)
        self.write_pos = 0
        self.lock = threading.Lock()
        self.stream = None
        self.running = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)

        samples = indata[:, 0]

        with self.lock:
            n = len(samples)

            if n >= self.size:
                self.buffer[:] = samples[-self.size:]
                self.write_pos = 0
                return

            end = self.write_pos + n

            if end <= self.size:
                self.buffer[self.write_pos:end] = samples
            else:
                first = self.size - self.write_pos
                self.buffer[self.write_pos:] = samples[:first]
                self.buffer[:n-first] = samples[first:]

            self.write_pos = end % self.size

    def resume(self):
        if self.running:
            return

        self.running = True

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
        )

        self.stream.start()

    def pause(self):
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get(self, length_ms):
        n = int(self.sample_rate * length_ms / 1000)

        with self.lock:
            if n >= self.size:
                n = self.size

            # Read backwards from current write position.
            start = self.write_pos - n

            if start >= 0:
                return self.buffer[start:self.write_pos].copy()

            return np.concatenate(
                (
                    self.buffer[start:],
                    self.buffer[:self.write_pos],
                )
            )