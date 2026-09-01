#!/usr/bin/env python3

from constants import SAMPLE_RATE, DEVICE, LONG_REPORT_NAME, BATCH_SIZE, VALIDATION_SPLIT, \
    DENOISER_LEARNING_RATE
from constrained_generation import run_constrained_generation_experiment
from dataset import AudioDataset
from helpers import collate
from loops import transcribe, train, validate
import os

import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# Reproducibility
torch.manual_seed(42)


MODEL_TO_LOAD = None

CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


class Denoiser(nn.Module):
    def __init__(self, channels=(80, 64, 32, 16)):
        super().__init__()

        self.dropout = nn.Dropout1d(p=0.1)

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
        self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []

        w = x

        # Encoder
        for layer in self.encoder:
            w = self.relu(layer(w))
            w = self.dropout(w)
            skips.append(w)
            w = self.pool(w)

        # Bottleneck
        w = self.relu(self.bottleneck(w))
        w = self.dropout(w)

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

        return self.tanh(self.final(w)) + x


def main():



    writer = SummaryWriter(log_dir=f"runs/experiment_{len(os.listdir("runs")) + 1}")

    # base_model = "./whisper-domain"
    # base_model = "openai/whisper-tiny.en"
    base_model = "openai/whisper-small.en"
    processor = WhisperProcessor.from_pretrained(
        base_model,
        language="english",
        task="transcribe"
    )
    whisper = WhisperForConditionalGeneration.from_pretrained(base_model).to(DEVICE)
    whisper.eval()  # Keep whisper in eval mode since we're not training it

    denoiser = Denoiser().to(DEVICE)
    denoiser.add_input = True
    denoiser.eval()
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=DENOISER_LEARNING_RATE)
    if MODEL_TO_LOAD is not None:
        saved_object = torch.load(f"released_models/{MODEL_TO_LOAD}/{MODEL_TO_LOAD}_denoiser.pt", map_location=torch.device(DEVICE))
        optimizer.load_state_dict(saved_object["optimizer"])
        denoiser.load_state_dict(saved_object["model"])
    else:
        optimizer = None

    # Prep data
    full_dataset = AudioDataset()

    training_size = 800
    total_samples = len(full_dataset)
    print(f"Loaded {total_samples} total samples")
    validation_count = total_samples - training_size
    if validation_count / total_samples < VALIDATION_SPLIT:
        raise Exception(f"training size of {training_size} with a total of {total_samples} samples leaves only {100 * validation_count / total_samples:.1f}% for validation, but you wanted at least {100 * VALIDATION_SPLIT:.0f}%")
    
    train_dataset = Subset(full_dataset, range(training_size))
    val_dataset = Subset(full_dataset, range(training_size, len(full_dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # Train logit processors
    # run_constrained_generation_experiment(train_loader, val_loader, denoiser, processor, whisper, DEVICE)

    # Validate
    """
    val_loss, average_wer = validate(denoiser, val_loader, processor, whisper, DEVICE, 0)
    print(f"Average WER at top level validation: {100 * average_wer:.1f}%")
    """

    # Train
    train(train_loader, val_loader, denoiser, processor, whisper, optimizer, writer=writer)
    # fine_tune_whisper(train_loader, val_loader, whisper, processor, DEVICE)


    # Print a transcripting utilizing the denoiser
    path_to_transcribe = f"../backend/data/{LONG_REPORT_NAME}.wav"
    transcribe(denoiser, path_to_transcribe, processor, whisper, DEVICE)


if __name__ == "__main__":
    main()
