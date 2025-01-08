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
import torch.nn.functional as F


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

        # I don't know why but depending on the config, this is a list of len(1)
        tokens = tokens[0]

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

        """
        # Freeze parameters except decoder and lm_head
        for name, param in self.model.named_parameters():
            if name.startswith("model.encoder"):

                param.requires_grad = False

                for layer in ["28", "29", "30", "31", "model.encoder.layer_norm"]:
                    if layer in name:
                        param.requires_grad = True
                        break

                if param.requires_grad:
                    print("Unfreezing...", name)
                else:
                    print("Freezing...", name)

            else:
                param.requires_grad = True
                print("Unfreezing...", name)
        """

        # Freeze parameters except decoder and lm_head
        for name, param in self.model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

            for layer in ["28", "29", "30", "31"]:
                if layer in name:
                    print("Unfreezing...", name)
                    param.requires_grad = True

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

    def get_optimizer(self, lr=1e-4, weight_decay=1e-2):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
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
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # Training loop
            for batch in progress_bar:
                if not batch:
                    continue

                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_features=input_features, labels=labels)

                # loss = outputs.loss
                loss = self.custom_loss(outputs.logits, labels)

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
                    # f.write(f"VAL_PREDICTIONS Epoch {epoch + 1}: {json.dumps(predictions)}\n")

                # Update progress bar with validation loss
                progress_bar.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}"
                )

            # Print summary for the epoch
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            tqdm.write(f"  Train Loss: {avg_train_loss:.4f}")
            if (epoch + 1) % eval_every == 0:
                tqdm.write(f"  Validation Loss: {val_loss:.4f}")

    # TODO: Consider weighted loss
    def custom_loss(self, logits, targets, repetition_penalty=0.1):
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.tokenizer.pad_token_id)
        batch_size, seq_len, vocab_size = logits.size()
        probs = F.softmax(logits, dim=-1)
        token_counts = torch.sum(probs, dim=1)
        repetition_loss = torch.sum(token_counts ** 2) / (batch_size * seq_len)
        return ce_loss + repetition_penalty * repetition_loss

    def predict(self, input_mel, max_length=100, num_beams=1):
        self.model.eval()
        with torch.no_grad():
            input_mel = input_mel.to(self.device).unsqueeze(0)  # Add batch dimension
            eos_token_id = self.tokenizer["EOS_None"]
            outputs = self.model.generate(
                input_features=input_mel,
                max_length=max_length,
                eos_token_id=eos_token_id,
                early_stopping=True,
                num_beams=num_beams,
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

    if os.path.exists("tokenizer.pkl"):
        tokenizer = torch.load("tokenizer.pkl")
        return tokenizer

    """
    tokenizer_config = TokenizerConfig(special_tokens=["PAD_None", "BOS_None", "EOS_None", "MASK_None"],
                                       num_velocities=16, use_chords=True, use_programs=True, use_durations=True,
                                       use_time_signatures=True, use_pitch_bends=True, use_rests=True, use_tempo=True,
                                       use_sustain_pedals=True)
    """

    # Temporary simplification
    tokenizer_config = TokenizerConfig(
        special_tokens=["PAD_None", "BOS_None", "EOS_None", "MASK_None"],
        num_velocities=8,
        use_chords=False,
        use_programs=False,
        use_durations=True,
        use_time_signatures=False,
        use_pitch_bends=False,
        use_rests=False,
        use_tempo=False,
        use_sustain_pedals=False
    )

    tokenizer = REMI(tokenizer_config)
    train_midis = list(Path("synthetic_data/train").glob("*.mid"))
    tokenizer.train(vocab_size=300, model="BPE", files_paths=train_midis)
    torch.save(tokenizer, "tokenizer.pkl")

    print("Special Token IDs:")
    print(f"BOS: {tokenizer['BOS_None']}")
    print(f"EOS: {tokenizer['EOS_None']}")
    print(f"MASK: {tokenizer['MASK_None']}")
    print(f"PAD: {tokenizer['PAD_None']}")
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

