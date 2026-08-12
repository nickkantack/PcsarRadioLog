#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# Configuration
VALIDATION_SPLIT = 0.2  # 20% for validation, 80% for training
writer = SummaryWriter(log_dir=f"runs/experiment_{len(os.listdir("runs")) + 1}")

CHECKPOINT_PREFIX = "additive-end_"
EPOCH_TO_LOAD = None
EPOCHS_TO_TRAIN = 1

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Denoiser(nn.Module):
    def __init__(self, channels=(80, 64, 32, 16)):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])

        # Bottleneck
        self.bottleneck = nn.Conv1d(
            channels[-1],
            channels[-1],
            kernel_size=3,
            padding=1,
        )

        self.add_input = True

        # Decoder
        #
        # Encoder outputs (skip connections):
        # [64, 32, 16]
        #
        # Decoder:
        # (16+16)->16
        # (16+32)->32
        # (32+64)->64
        #
        skip_channels = list(channels[1:])
        skip_channels.reverse()

        decoder = []
        current_channels = channels[-1]

        for skip_ch in skip_channels:
            decoder.append(
                nn.Conv1d(
                    current_channels + skip_ch,
                    skip_ch,
                    kernel_size=3,
                    padding=1,
                )
            )
            current_channels = skip_ch

        self.decoder = nn.ModuleList(decoder)

        # Restore original feature dimension
        self.final = nn.Conv1d(channels[1], channels[0], kernel_size=1)

        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skips = []

        w = x

        # Encoder
        for layer in self.encoder:
            w = self.relu(layer(w))
            skips.append(w)
            w = self.pool(w)

        # Bottleneck
        w = self.relu(self.bottleneck(w))

        # Decoder
        for layer, skip in zip(self.decoder, reversed(skips)):
            w = self.up(w)

            # Match temporal dimensions
            if w.size(-1) > skip.size(-1):
                w = w[..., :skip.size(-1)]
            elif w.size(-1) < skip.size(-1):
                skip = skip[..., :w.size(-1)]

            w = torch.cat((w, skip), dim=1)
            w = self.relu(layer(w))

        """
        By making the output additive on the input, the auto-encoder part of the 
        U-Net has the option to decline to make large changes to the input without
        having to do the difficult work of learning to reconstruct the input
        from a low dimensional representation. Pretraining then should teach the
        U-Net to NOT change the input, so that when the training task shifts to
        transcription accuracy the U-Net explores small changes to the input
        over large ones. In theory, the U-Net can spend all of its degrees of
        freedom learning denoising rather than audio reconstruction.
        """

        # TODO make it configurable whether the input is added or not. Train
        # with the input NOT added first, the U-Net gets a good internal representation,
        # then train with the input added, wait for reconstruction loss to be
        # significantly smaller than it was without the added input, then switch
        # to training for transcription accuracy.

        if self.add_input:
            return self.final(w) + x
        else:
            """
            Idea: When training with self.add_input = True, compute loss by comparing
            the Denoiser's output to the original audio WITH IT'S MEAN SUBTRACTED.
            The idea is that the Denoiser still has to reproduce the relative shape
            of the audio, so it learns useful representations, but it also biases
            towards a zero mean change on the audio, which means that when 
            self.add_input flips to True, the Denoiser output suddenly exaggerates
            the scale of x but doesn't shift the values, so training immediately
            starts with weights moving relative to one another rather than all
            weights racing down towards zero (they still race towards zero initially
            under this idea, but something feels better about having the Denoiser
            output pixels on both sides of the correct answer rather than all too
            large. Maybe this doesn't change much.
            """
            return self.final(w)


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


MEL_CHUNK = 200  # 2 seconds ≈ 200 frames
def chunk_mel(mel, chunk_size=MEL_CHUNK):
    """
    mel: (1, 80, T)

    returns list of (1,80,chunk_size)
    """

    chunks = []

    T = mel.size(-1)

    for start in range(0, T, chunk_size):
        c = mel[..., start:start + chunk_size]

        if c.size(-1) < chunk_size:
            c = torch.nn.functional.pad(
                c,
                (0, chunk_size - c.size(-1))
            )

        chunks.append(c)

    return chunks, T


def validate(model, val_loader, processor, whisper, device, global_step):
    """Run validation and return average validation loss"""
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for waveforms, texts in val_loader:
            for waveform, transcript in zip(waveforms, texts):
                features = processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )

                mel = features.input_features.to(DEVICE)

                chunks, original_length = chunk_mel(mel)

                processed = []

                for chunk in chunks:
                    processed.append(model(chunk))

                combined_mel = torch.cat(processed, dim=-1)

                # Remove padding added by chunk_mel()
                combined_mel = combined_mel[..., :original_length]

                labels = processor.tokenizer(
                    transcript,
                    return_tensors="pt",
                ).input_ids.to(DEVICE)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = whisper(
                        input_features=combined_mel,
                        labels=labels,
                    )
                    loss = outputs.loss

                total_val_loss += loss.item()
                num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    model.train()
    return avg_val_loss


def train(denoiser, optimizer, processor, whisper):

    # Print model summary early
    """
    denoiser_temp = Denoiser()
    summary(denoiser_temp, input_size=(1, 80, 2000))  # Assuming ~1500 time steps for 2-second audio
    del denoiser_temp
    print("\n")
    """

    # freeze whisper weights while preserving autograd
    for p in whisper.parameters():
        p.requires_grad = False

    starting_epoch = EPOCH_TO_LOAD if EPOCH_TO_LOAD is not None else 0
    
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

    global_step = 0
    for epoch in range(starting_epoch, starting_epoch + EPOCHS_TO_TRAIN):
        denoiser.train()
        batch_num = 0
        for waveforms, texts in loader:
            optimizer.zero_grad()
            total_loss = 0.0
            total_whisper_loss = 0.0
            total_mse_loss = 0.0
            
            for waveform, transcript in zip(waveforms, texts):
                features = processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )

                mel = features.input_features.to(DEVICE)

                # Compute the mask of where audio actually is within the
                # 30 second mel.
                is_audio_mask = torch.where(
                    mel > torch.min(mel), 
                    torch.ones_like(mel), 
                    torch.zeros_like(mel)
                )
                is_audio_mask.to(DEVICE)

                chunks, original_length = chunk_mel(mel)

                processed = []

                for chunk in chunks:
                    # Compute denoised output
                    denoised_chunk = denoiser(chunk)
                    processed.append(denoised_chunk)
                    

                combined_mel = torch.cat(processed, dim=-1)

                # Remove padding added by chunk_mel()
                combined_mel = combined_mel[..., :original_length]

                mse_loss = torch.nn.functional.mse_loss(combined_mel * is_audio_mask, mel * is_audio_mask)

                """
                labels = processor.tokenizer(
                    transcript,
                    return_tensors="pt",
                ).input_ids.to(DEVICE)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = whisper(
                        input_features=combined_mel,
                        labels=labels,
                    )
                    whisper_loss = outputs.loss
                """

                # Choose loss function
                # loss = whisper_loss
                loss = mse_loss

                global_step += 1
                batch_num += 1
                writer.add_scalar("Loss/Train", loss, global_step)

            # Backpropagate the accumulated loss for the batch
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"\repoch={epoch + 1}/{starting_epoch + EPOCHS_TO_TRAIN} batch={batch_num}/{len(loader)} loss={loss.item():.3f}", end="")
        print()

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
            f"{CHECKPOINT_PREFIX}{epoch + 1}.pt",
        )


def denoise_to_wav(model, input_wav, output_wav, processor, device):
    model.eval()

    waveform, sr = torchaudio.load(input_wav)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    waveform = waveform.mean(0)

    with torch.no_grad():
        mel = processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device)

        denoised = model(mel).squeeze(0).cpu()  # (80, 3000)

        plt.imshow(denoised)
        plt.show()

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=201,
        n_mels=80,
        sample_rate=SAMPLE_RATE,
    )

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=400,
        hop_length=160,
    )

    linear_spec = inverse_mel(denoised)
    audio = griffin(linear_spec)

    torchaudio.save(output_wav, audio.unsqueeze(0), SAMPLE_RATE)

    model.train()


def transcribe(model, input_wav, processor, whisper, device):

    model.eval()

    waveform, sr = torchaudio.load(input_wav)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    waveform = waveform.mean(0)

    with torch.no_grad():
        mel = processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device)

        fig, axes = plt.subplots(2)
        axes[0].imshow(mel.clone().detach().cpu().squeeze())
        mel = model(mel)
        axes[1].imshow(mel.clone().detach().cpu().squeeze())
        plt.show()

        generated_ids = whisper.generate(
                input_features=mel
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        print(text)



def main():

    base_model = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(base_model)
    whisper = WhisperForConditionalGeneration.from_pretrained(base_model).to(DEVICE)
    whisper.eval()  # Keep whisper in eval mode since we're not training it

    denoiser = Denoiser().to(DEVICE)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)
    if EPOCH_TO_LOAD is not None:
        saved_object = torch.load(f"{CHECKPOINT_PREFIX}{EPOCH_TO_LOAD}.pt")
        denoiser.load_state_dict(saved_object["model"])
        optimizer.load_state_dict(saved_object["optimizer"])

    # Train
    train(denoiser, optimizer, processor, whisper)

    # Print a transcripting utilizing the denoiser
    path_to_transcribe = "../backend/data/" + list(filter(lambda x: x.endswith(".wav"), os.listdir("../backend/data")))[0]
    print(path_to_transcribe)
    transcribe(denoiser, path_to_transcribe, processor, whisper, DEVICE)


if __name__ == "__main__":
    main()