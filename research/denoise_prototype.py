#!/usr/bin/env python3

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# Configuration
VALIDATION_SPLIT = 0.2  # 20% for validation, 80% for training
writer = SummaryWriter(log_dir=f"runs/experiment_{len(os.listdir("runs")) + 1}")

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

    def __init__(self, filenames_sans_extensions=None):
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

    def get_total_audio_seconds(self):
        """Return the total duration of all audio files in seconds."""
        total_seconds = 0.0

        for name in self.filenames_sans_extensions:
            path = f"../backend/data/{name}.wav"
            waveform, sample_rate = torchaudio.load(path)
            total_seconds += waveform.shape[1] / sample_rate

        return total_seconds


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


def validate(model, val_loader, processor, whisper, device, global_step):
    """Run validation and return average validation loss"""
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for waveforms, texts in val_loader:
            for waveform, transcript in zip(waveforms, texts):
                # Split entire waveform into chunks
                chunks = chunk_audio(waveform)
                
                # Process all chunks through denoiser
                processed_mels = []
                for chunk in chunks:
                    # Whisper feature extraction
                    features = processor.feature_extractor(
                        chunk.numpy(),
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    
                    mel = features.input_features.to(device)
                    # Apply denoising
                    denoised_mel = model(mel)
                    processed_mels.append(denoised_mel)
                
                # Concatenate processed mel spectrograms along time dimension
                if len(processed_mels) > 1:
                    combined_mel = torch.cat(processed_mels, dim=2)  # Concatenate along time axis
                else:
                    combined_mel = processed_mels[0]
                
                # Transcribe the entire denoised audio
                labels = processor.tokenizer(transcript, return_tensors="pt").input_ids.to(device)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = whisper(input_features=combined_mel, labels=labels)
                    loss = outputs.loss
                
                total_val_loss += loss.item()
                num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    model.train()
    return avg_val_loss


def main():

    base_model = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(base_model)
    whisper = WhisperForConditionalGeneration.from_pretrained(base_model).to(DEVICE)
    whisper.eval()  # Keep whisper in eval mode since we're not training it

    # freeze whisper weights while preserving autograd
    for p in whisper.parameters():
        p.requires_grad = False

    denoiser = Denoiser().to(DEVICE)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-5)
    
    # Create single dataset and split into train/validation
    full_dataset = AudioDataset()
    total_samples = len(full_dataset)
    validation_count = int(total_samples * VALIDATION_SPLIT)
    train_count = total_samples - validation_count
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_count, validation_count],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    epochs = 10
    global_step = 0
    for epoch in range(epochs):
        denoiser.train()
        for waveforms, texts in loader:
            optimizer.zero_grad()
            total_loss = 0.0
            
            for waveform, transcript in zip(waveforms, texts):
                # Split entire waveform into chunks
                chunks = chunk_audio(waveform)
                
                # Process all chunks through denoiser
                processed_mels = []
                for chunk in chunks:
                    # Whisper feature extraction
                    features = processor.feature_extractor(
                        chunk.numpy(),
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    
                    mel = features.input_features.to(DEVICE)
                    # Apply denoising (keeping gradients)
                    denoised_mel = denoiser(mel)
                    processed_mels.append(denoised_mel)
                
                # Concatenate processed mel spectrograms along time dimension
                if len(processed_mels) > 1:
                    combined_mel = torch.cat(processed_mels, dim=2)  # Concatenate along time axis
                else:
                    combined_mel = processed_mels[0]
                
                # Transcribe the entire denoised audio
                labels = processor.tokenizer(transcript, return_tensors="pt").input_ids.to(DEVICE)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = whisper(input_features=combined_mel, labels=labels)
                    loss = outputs.loss
                
                total_loss += loss
                global_step += 1
                writer.add_scalar("Loss/train", loss, global_step)

            # Backpropagate the accumulated loss for the batch
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"epoch={epoch} " f"loss={total_loss.item():.4f}")

        # Run validation at the end of each epoch
        print(f"Running validation for epoch {epoch}...")
        val_loss = validate(denoiser, val_loader, processor, whisper, DEVICE, global_step)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        print(f"Validation loss: {val_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": denoiser.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f"checkpoint_{epoch}.pt",
        )


if __name__ == "__main__":
    main()