import time
from audio import get_audio


def whisper_inference(model, processor, audio, measure_time=True):

    # We expect audio to be pre-processed, but this is a fallback for if a path is passed.
    if isinstance(audio, str):
        audio = get_audio(audio, processor=processor)

    if measure_time: start = time.time()

    # This is where the bulk of the compute happens; running through the Whisper model.
    outputs = model.generate(
        audio,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    if measure_time: end = time.time()
    if measure_time: print("Inference took", round(end - start, 2), "seconds")

    # Take the transcription for reference.
    transcription = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    # Extract hidden states immediately before the final linear/softmax layer for each token.
    hidden_states = outputs.decoder_hidden_states  # (time steps, layer states)
    final_hidden_states = []

    # Here each timestep represents a token in the output sequence, with the state updated after the model generates each token.
    # Because Transformer models manage their own self-attention, the hidden states change as it continues predicting.
    for timestep_hidden_states in hidden_states[1:]:

        # Taking the last layer hidden state for each timestep and squeezing the batch dimension for a single sequence.
        hidden_state = timestep_hidden_states[-1].squeeze(0).squeeze(0)
        final_hidden_states.append(hidden_state)

    return transcription, final_hidden_states
