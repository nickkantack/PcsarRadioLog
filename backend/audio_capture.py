import threading

import numpy as np
import sounddevice as sd
import torch
import torchaudio


class AudioCapture:
    def __init__(
        self,
        length_ms,
        input_sample_rate=44100,
        output_sample_rate=16000,
        device=None,
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.device = device

        # Buffer size is based on the rate at which samples
        # actually arrive from the sound device.
        self.size = int(
            input_sample_rate * length_ms / 1000
        )

        self.buffer = np.zeros(
            self.size,
            dtype=np.float32,
        )

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
                self.buffer[:n - first] = samples[first:]

            self.write_pos = end % self.size

    def resume(self):
        if self.running:
            return

        self.running = True

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.input_sample_rate,
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
        # Number of samples corresponding to length_ms
        # at the ACTUAL capture rate.
        n = int(
            self.input_sample_rate * length_ms / 1000
        )

        with self.lock:
            if n >= self.size:
                n = self.size

            start = self.write_pos - n

            if start >= 0:
                audio = self.buffer[
                    start:self.write_pos
                ].copy()
            else:
                audio = np.concatenate(
                    (
                        self.buffer[start:],
                        self.buffer[:self.write_pos],
                    )
                )

        # Convert from hardware rate to Whisper rate.
        if self.input_sample_rate != self.output_sample_rate:
            waveform = torch.from_numpy(audio).unsqueeze(0)

            waveform = torchaudio.functional.resample(
                waveform,
                self.input_sample_rate,
                self.output_sample_rate,
            )

            audio = waveform.squeeze(0).numpy()

        return audio
