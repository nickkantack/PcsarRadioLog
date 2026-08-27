from constants import SAMPLE_RATE
import json
import os
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):

    def __init__(self, filenames_sans_extensions=None, size=None):
        if filenames_sans_extensions is not None:
            self.filenames_sans_extensions = filenames_sans_extensions
        else:
            self.filenames_sans_extensions = []
            candidates = set()
            candidates_with_extensions = set(os.listdir("../backend/data"))
            for filename in candidates_with_extensions:
                candidates.add(filename.split(".")[0])

            for candidate in candidates:
                if f"{candidate}.wav" in candidates_with_extensions and f"{candidate}.txt" in candidates_with_extensions:
                    with open(f"../backend/data/{candidate}.txt", "r") as file:
                        properties = json.loads(file.read())
                        if "label" in properties and properties["label"] != "X":
                            self.filenames_sans_extensions.append(candidate)
            
            self.filenames_sans_extensions.sort()

            if size is not None:
                self.filenames_sans_extensions = self.filenames_sans_extensions[:size]
        

    def __len__(self):
        return len(self.filenames_sans_extensions)


    def __getitem__(self, idx):
        audio, sr = torchaudio.load(f"../backend/data/{self.filenames_sans_extensions[idx]}.wav")
        text = ""
        with open(f"../backend/data/{self.filenames_sans_extensions[idx]}.txt", "r") as file:
            text = json.loads(file.read())["label"]
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.mean(0)
        return audio, text


    def get_total_audio_seconds(self):
        """Return the total duration of all audio files in seconds."""
        total_seconds = 0.0

        for name in self.filenames_sans_extensions:
            path = f"../backend/data/{name}.wav"
            waveform, sample_rate = torchaudio.load(path)
            total_seconds += waveform.shape[1] / sample_rate

        return total_seconds