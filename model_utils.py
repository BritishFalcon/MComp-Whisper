from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from miditok import REMI, TokenizerConfig
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperFeatureExtractor
from torch import nn
from tqdm import tqdm
import os
import json


class MusicDataset(Dataset):
    def __init__(self, audio_dir, midi_dir, tokenizer, feature_extractor, sample_rate=16000):
        self.audio_dir = Path(audio_dir)
        self.midi_dir = Path(midi_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor

        audio_files = {f.stem: f for f in self.audio_dir.glob("*.wav")}
        midi_files = {f.stem: f for f in self.midi_dir.glob("*.mid")}
        self.common_files = [
            (audio_files[stem], midi_files[stem])
            for stem in audio_files.keys() & midi_files.keys()
        ]

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        audio_path, midi_path = self.common_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        mel = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features[0].squeeze(0)
        tokens = self.tokenizer.encode(midi_path)
        target_ids = [self.tokenizer["BOS_None"]] + tokens.ids + [self.tokenizer["EOS_None"]]
        return {
            "input_mel": mel,
            "target_ids": target_ids,
            "file_name": audio_path.name  # Include the file name
        }



def CustomCollate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    max_mel_length = 3000
    input_mel = []
    file_names = []  # Handy for manual comparison later

    for item in batch:
        mel = item["input_mel"]
        file_names.append(item["file_name"])  # Add file name to the list
        if mel.shape[1] > max_mel_length:
            mel = mel[:, :max_mel_length]
        else:
            pad_length = max_mel_length - mel.shape[1]
            mel = torch.nn.functional.pad(mel, (0, pad_length))
        input_mel.append(mel)

    input_mel = torch.stack(input_mel)

    pad_token_id = 0
    max_target_len = max(len(item["target_ids"]) for item in batch)
    target_ids = torch.full((len(batch), max_target_len), pad_token_id, dtype=torch.long)

    for i, item in enumerate(batch):
        target_len = len(item["target_ids"])
        target_ids[i, :target_len] = torch.tensor(item["target_ids"], dtype=torch.long)

    return {
        "input_features": input_mel,
        "labels": target_ids,
        "file_names": file_names  # Include file names in the batch
    }


class WhisperREMIModel:
    def __init__(
        self,
        tokenizer,
        feature_extractor,
        model_name="openai/whisper-large-v3-turbo",
        device=None,
        checkpoint_path=None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        # Initialize model configuration
        self.config = WhisperConfig.from_pretrained(model_name)
        self.config.vocab_size = len(self.tokenizer)
        self.config.pad_token_id = self.tokenizer["PAD_None"]
        self.config.eos_token_id = self.tokenizer["EOS_None"]
        self.config.bos_token_id = self.tokenizer["BOS_None"]
        self.config.decoder_start_token_id = self.tokenizer["BOS_None"]
        self.config.mask_token_id = self.tokenizer["MASK_None"]

        # Initialize model
        self.model = WhisperForConditionalGeneration(self.config)
        self.model.model.decoder.embed_tokens = nn.Embedding(len(self.tokenizer), self.config.d_model)
        nn.init.xavier_uniform_(self.model.model.decoder.embed_tokens.weight)
        self.model.lm_head = nn.Linear(self.config.d_model, len(self.tokenizer), bias=False)
        nn.init.xavier_uniform_(self.model.lm_head.weight)
        self.model.to(self.device)

        # Freeze parameters except decoder and lm_head
        for name, param in self.model.named_parameters():
            if "decoder" not in name and "lm_head" not in name:
                param.requires_grad = False

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Loaded model checkpoint from {checkpoint_path}")

    def save_checkpoint(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

    def get_optimizer(self, lr=1e-4):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def train(
            self,
            train_loader,
            val_loader,  # New DataLoader for validation data
            optimizer,
            num_epochs=5,
            checkpoint_dir="checkpoints",
            log_file="training_log.txt",  # Log file for losses and predictions
            save_every_epoch=True,
            eval_every=1  # Evaluate and log validation loss every `eval_every` epochs
    ):
        self.model.train()
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Prepare logging
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("")  # Clear or create file

        for epoch in range(num_epochs):
            total_train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")

            # Training loop
            for batch in progress_bar:
                if not batch:
                    continue

                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader)

            # Log training loss
            with open(log_file, "a") as f:
                f.write(f"TRAIN_LOSS Epoch {epoch + 1}: {avg_train_loss:.4f}\n")

            # Save model checkpoint
            if save_every_epoch:
                checkpoint_path = Path(checkpoint_dir) / f"model_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path)

            # Evaluate on validation set
            if (epoch + 1) % eval_every == 0:
                val_loss, predictions = self.evaluate(val_loader)
                with open(log_file, "a") as f:
                    f.write(f"VAL_LOSS Epoch {epoch + 1}: {val_loss:.4f}\n")
                    # Log predictions in JSON format for easier parsing later
                    f.write(f"VAL_PREDICTIONS Epoch {epoch + 1}: {json.dumps(predictions)}\n")

    def predict(self, input_mel, max_length=100, eos_token_id=None):
        self.model.eval()
        with torch.no_grad():
            input_mel = input_mel.to(self.device).unsqueeze(0)  # Add batch dimension
            if eos_token_id is None:
                eos_token_id = self.tokenizer.eos_token_id
            outputs = self.model.generate(
                input_features=input_mel,
                max_length=max_length,
                eos_token_id=eos_token_id,
                early_stopping=True
            )
            return outputs[0].tolist()

    def evaluate(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        predictions = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", unit="batch"):
                if not batch:
                    continue

                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Collect predictions for logging
                preds = self.model.generate(input_features=input_features)
                for pred, label, file_name in zip(preds, labels, batch["file_names"]):
                    predictions.append({
                        "file_name": file_name,
                        "predicted_ids": pred.tolist(),
                        "target_ids": label.tolist()
                    })

        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss, predictions


def initialize_tokenizer():
    tokenizer_config = TokenizerConfig(special_tokens=["PAD_None", "BOS_None", "EOS_None", "MASK_None"],
                             num_velocities=16, use_chords=True, use_programs=True, use_durations=True,
                             use_time_signatures=True, use_pitch_bends=True, use_rests=True, use_tempo=True,
                             use_sustain_pedals=True)
    tokenizer = REMI(tokenizer_config)
    print("Special Token IDs:")
    print(f"BOS: {tokenizer['BOS_None']}")
    print(f"EOS: {tokenizer['EOS_None']}")
    print(f"PAD: {tokenizer.pad_token_id}")
    print(f"MASK: {tokenizer['MASK_None']}")
    return tokenizer


def initialize_feature_extractor(model_name="openai/whisper-large-v3-turbo"):
    return WhisperFeatureExtractor.from_pretrained(model_name)


def create_data_loader(audio_dir, midi_dir, tokenizer, feature_extractor, batch_size=16, sample_rate=16000, shuffle=True):
    dataset = MusicDataset(audio_dir, midi_dir, tokenizer, feature_extractor, sample_rate=sample_rate)
    return DataLoader(dataset=dataset, collate_fn=CustomCollate, batch_size=batch_size, shuffle=shuffle)


def create_validation_loader(
    audio_dir,
    midi_dir,
    tokenizer,
    feature_extractor,
    batch_size=16,
    sample_rate=16000
):
    return create_data_loader(
        audio_dir=audio_dir,
        midi_dir=midi_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        sample_rate=sample_rate,
        shuffle=False
    )

