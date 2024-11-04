"""
This module will serve to pre-process the dataset we will eventually
use to train the Transformer model. Here, we will pass in our training
dataset of the music itself, run each piece through Whisper, then associate
this set of hidden states with the corresponding MIDI track, giving us a
ground truth and input pair for our model.
"""
import os.path

# TODO: Learn how to convert from MIDI to chosen tokenizer output, allowing a same-structure
#       comparison between the ground truth and the model output.

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from inference import whisper_inference
from audio import get_audio

from pathlib import Path
import pickle


class Datapoint:
    def __init__(self, source_filename, tokenized_midi, whisper_hidden_states, transcription):
        self.source_filename = source_filename
        self.target = tokenized_midi

        self.input = whisper_hidden_states
        self.transcription = transcription


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the pre-trained Whisper model and processor
model_name = "openai/whisper-large"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

model.to(device)

for midi_file in Path("data/").rglob("*.mid"):
    wav_file = f"{midi_file.stem}.wav"
    assert os.path.exists(wav_file), f"Missing WAV file for {midi_file}"

    audio = get_audio(wav_file, processor=processor, device=device)
    transcript, hidden_states = whisper_inference(model=model, processor=processor, audio=audio, measure_time=True)

    # TODO: Formulate method to convert MIDI to our tokenizer's format.
    midi_track = None

    datapoint = Datapoint(source_filename=midi_file,
                          tokenized_midi=midi_track,
                          whisper_hidden_states=hidden_states,
                          transcription=transcript)

    pickle.dump(datapoint, open(f"{midi_file.stem}.pkl", "wb"))
