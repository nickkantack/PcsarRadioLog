from constants import SAMPLE_RATE, DEVICE, writer, LONG_REPORT_NAME, DENOISER_LEARNING_RATE, \
    BATCH_SIZE, VALIDATION_SPLIT, EPOCHS_TO_TRAIN, CHECKPOINT_PREFIX
from dataset import AudioDataset
from helpers import chunk_mel, wer, collate
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchaudio

def validate(model, val_loader, processor, whisper, device, global_step, logits_processors=None):
    """Run validation and return average validation loss"""
    if model:
        model.eval()
    total_val_loss = 0.0
    average_wer = 0
    num_batches = 0

    whisper.eval()
    
    batch_index = 0
    with torch.no_grad():
        for waveforms, texts in val_loader:
            batch_index += 1
            for waveform, transcript in zip(waveforms, texts):
                features = processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )

                mel = features.input_features.to(DEVICE)

                if model is None:
                    combined_mel = mel
                else:

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

                outputs = whisper(
                    input_features=combined_mel,
                    labels=labels,
                )
                loss = outputs.loss

                total_val_loss += loss.item()

                generated_ids = whisper.generate(
                        input_features=combined_mel,
                        num_beams=5,
                        do_sample=False,
                        early_stopping=True,
                        logits_processor=logits_processors if logits_processors is not None else []
                    )

                predicted_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]

                average_wer += wer(transcript, predicted_text)
                num_batches += 1

            print(f"\rValidated {batch_index}/{len(val_loader)} batches", end="")
        print()
    
    total_val_loss /= num_batches
    average_wer /= num_batches

    # Generate an example prediction/ground truth pair to write in tensorboard
    filename_no_extension = LONG_REPORT_NAME
    with open(f"../backend/data/{filename_no_extension}.txt", "r") as file:
        properties = json.loads(file.read())
        prediction = transcribe(model, f"../backend/data/{filename_no_extension}.wav", processor, whisper, device, logits_processors)
        reference = properties["label"]
        writer.add_text(
            "Examples/Transcript",
            f"**Reference:** {reference}\n\n"
            f"**Prediction:** {prediction}",
            global_step,
        )

    if model:
        model.train()

    return total_val_loss, average_wer


def transcribe(denoiser, input_wav, processor, whisper, device, logits_processors=None):

    if denoiser:
        denoiser.eval()

    whisper.eval()

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

        """
        fig, axes = plt.subplots(2)
        axes[0].imshow(mel.clone().detach().cpu().squeeze())
        """
        if denoiser:
            mel = denoiser(mel)
        """
        axes[1].imshow(mel.clone().detach().cpu().squeeze())
        plt.show()
        """

        generated_ids = whisper.generate(
                input_features=mel,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
                logits_processor=logits_processors if logits_processors is not None else []
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        print(text)
    
        if denoiser:
            denoiser.train()

        return text


def train(denoiser, processor, whisper, optimizer, do_normalize=False):

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

    # Create single dataset and split into train/validation
    full_dataset = AudioDataset(do_normalize=do_normalize)
    total_samples = len(full_dataset)
    validation_count = int(total_samples * VALIDATION_SPLIT)
    train_count = total_samples - validation_count
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_count, validation_count],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    global_step = 0
    denoiser.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(denoiser.parameters(), lr=DENOISER_LEARNING_RATE)
    for epoch in range(EPOCHS_TO_TRAIN):
        denoiser.train()
        batch_num = 0
        for waveforms, texts in loader:
            optimizer.zero_grad()

            batch_loss = 0.0
            
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

                labels = processor.tokenizer(
                    transcript,
                    return_tensors="pt",
                ).input_ids.to(DEVICE)

                outputs = whisper(
                    input_features=combined_mel,
                    labels=labels,
                )
                whisper_loss = outputs.loss

                # Choose loss function
                loss = 0 * mse_loss + whisper_loss

                batch_loss += loss

            global_step += 1

            batch_num += 1

            batch_loss = batch_loss / len(waveforms)
            writer.add_scalar("Loss/Train (batch average)", batch_loss, global_step)
            batch_loss.backward()
            optimizer.step()

            print(f"\repoch={epoch + 1}/{EPOCHS_TO_TRAIN} batch={batch_num}/{len(loader)} loss={loss.item():.3f}", end="")
        print()

        # Run validation at the end of each epoch
        print(f"Running validation for epoch {epoch}...")
        val_loss, average_wer = validate(denoiser, val_loader, processor, whisper, DEVICE, global_step)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("normalized WER/validation", average_wer, epoch)
        print(f"Validation loss: {val_loss:.4f}")

        if (epoch + 1) % 20 == 0 or epoch == EPOCHS_TO_TRAIN - 1:
            torch.save(
                {
                    "model": denoiser.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"{CHECKPOINT_PREFIX}{epoch + 1}ep.pt",
            )


def fine_tune_whisper(whisper, processor, device):

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
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    global_step = 0

    optimizer = torch.optim.AdamW(whisper.parameters(), lr=1e-6, weight_decay=0.01)

    num_epochs = 100

    for epoch in range(num_epochs):

        whisper.train()

        batch_num = 0
        for waveforms, texts in loader:

            optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            for waveform, transcript in zip(waveforms, texts):

                input_features = processor(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features

                labels = processor.tokenizer(
                    transcript,
                    return_tensors="pt",
                ).input_ids

                labels[labels == processor.tokenizer.pad_token_id] = -100

                input_features = input_features.to(device)
                labels = labels.to(device)

                outputs = whisper(
                    input_features=input_features,
                    labels=labels,
                )

                batch_loss += outputs.loss

            batch_loss = batch_loss / len(waveforms)
            batch_loss.backward()
            optimizer.step()

            batch_num += 1
            global_step += 1
            writer.add_scalar("Loss/Train", batch_loss, global_step)

            print(f"\repoch={epoch + 1}/{num_epochs} batch={batch_num}/{len(loader)}", end="")
        print()

        # Run validation at the end of each epoch
        print(f"Running validation for epoch {epoch}...")
        val_loss, average_wer = validate(None, val_loader, processor, whisper, DEVICE, global_step, full_dataset)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("WER/validation", average_wer, epoch)
        print(f"Validation loss: {val_loss:.4f}")

        # Save fine-tuned model
        whisper.save_pretrained("./whisper-domain")
        processor.save_pretrained("./whisper-domain")


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