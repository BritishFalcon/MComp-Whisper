import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from inference import whisper_inference
from audio import get_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the pre-trained Whisper model and processor
model_name = "openai/whisper-large"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

model.to(device)

"""
This is where the audio is processed before going into the model; specifically kept seperate to allow us to experiment
with any of the pre-processing steps we may wish to try as part of the research process. For now, this is simply
doing the pre-processing to put the audio in a state Whisper expects; 16kHz mono audio.
"""
audio = get_audio("speech/right.wav", processor=processor, device=device)
# Just using sample speech data to check all is working at present.

"""
This part is where the meat is - the forward-pass through Whisper for inference. In this, we're running Whisper as
usual for transcription, but also capturing the hidden states of the model as it continues token prediction. For our
work, we'll want to build a network which takes each set of hidden states and predicts our desired distribution.

Because we want the hidden states easily, we're using Transformers library rather than the bespoke Whisper library
provided by OpenAI as while I have messed with it before, it's not designed to be messed with. The HF library is more
built for development on top; good for us.
"""
transcript, hidden_states = whisper_inference(model=model, processor=processor, audio=audio, measure_time=True)

print("Transcription:")
print(transcript, end="\n\n")
print("Hidden States:")
[print(hidden_state.shape, hidden_state) for hidden_state in hidden_states]

# Below this is where out subsequent model head training will occur.
# Whisper -> States -> Result from States.
