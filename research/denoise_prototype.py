#!/usr/bin/env python3

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Conv1d(80, 80, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)


class AudioDataset(Dataset):

    def __init__(self):
        self.filenames_sans_extensions = []

        candidates = set()
        candidates_with_extensions = set(os.listdir("../backend/data"))
        for filename in candidates_with_extensions:
            candidates.add(filename.split(".")[0])

        for candidate in candidates:
            if f"{candidate}.wav" in candidates_with_extensions and f"{candidate}.txt" in candidates_with_extensions:
                self.filenames_sans_extensions.append(candidate)
        

    def __len__(self):
        return len(self.filenames_sans_extensions)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(f"../backend/data/{self.filenames_sans_extensions[idx]}.wav")
        text = ""
        with open(f"../backend/data/{self.filenames_sans_extensions[idx]}.txt", "r") as file:
            text = json.loads(file.read())["text"]
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.mean(0)
        return audio, text


def collate(batch):

    waveforms = []
    texts = []

    for wav, txt in batch:
        waveforms.append(wav)
        texts.append(txt)

    return waveforms, texts


def chunk_audio(wave):
    chunks = []
    pos = 0
    while pos < len(wave):
        c = wave[pos:pos + CHUNK_SAMPLES]
        if len(c) < CHUNK_SAMPLES:
            c = torch.nn.functional.pad(c, (0, CHUNK_SAMPLES - len(c)))
        chunks.append(c)
        pos += CHUNK_SAMPLES
    if len(chunks) == 0:
        chunks.append(torch.zeros(CHUNK_SAMPLES))
    return chunks


def main():

    base_model = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(base_model)
    whisper = WhisperForConditionalGeneration.from_pretrained(base_model).to(DEVICE)
    whisper.train()

    # freeze whisper weights while preserving autograd
    for p in whisper.parameters():
        p.requires_grad = False

    denoiser = Denoiser().to(DEVICE)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-5)
    dataset = AudioDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    epochs = 10
    for epoch in range(epochs):
        denoiser.train()
        for waveforms, texts in loader:
            optimizer.zero_grad()
            total_loss = 0.0
            for waveform, transcript in zip(waveforms, texts):
                chunks = chunk_audio(waveform)
                for chunk in chunks:
                    # Whisper feature extraction
                    features = processor.feature_extractor(
                        chunk.numpy(),
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )

                    mel = features.input_features.to(DEVICE)
                    mel = denoiser(mel)

                    labels = processor.tokenizer(transcript, return_tensors="pt").input_ids.to(DEVICE)

                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = whisper(input_features=mel, labels=labels)
                        loss = outputs.loss

                    total_loss += loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"epoch={epoch} " f"loss={total_loss.item():.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": denoiser.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"checkpoint_{epoch}.pt",
        )


if __name__ == "__main__":
    main()