import torch
import librosa


def get_audio(path, processor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_input, sample_rate = librosa.load(path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    return input_features
