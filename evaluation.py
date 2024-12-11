# evaluate.py

from model_utils import (
    initialize_tokenizer,
    initialize_feature_extractor,
    WhisperREMIModel,
    create_data_loader
)

def main():
    # Paths
    audio_dir = "synthetic_data/train"
    midi_dir = "synthetic_data/train"
    checkpoint_path = "checkpoints/model_epoch_5.pt"  # Update as needed

    # Initialize tokenizer and feature extractor
    tokenizer = initialize_tokenizer()
    feature_extractor = initialize_feature_extractor()

    # Initialize model (optionally loading from checkpoint)
    model = WhisperREMIModel(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        checkpoint_path=checkpoint_path  # Set to None if not loading
    )

    # Create data loader for evaluation
    data_loader = create_data_loader(
        audio_dir=audio_dir,
        midi_dir=midi_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=4,  # Adjust batch size as needed
        shuffle=False
    )

    # Process each batch
    for batch_idx, batch in enumerate(data_loader):
        if not batch:
            continue
        input_features = batch["input_features"].to(model.device)
        file_names = batch["file_names"]  # Retrieve file names
        labels = batch["labels"]

        # Perform prediction for the entire batch
        for i in range(input_features.size(0)):
            input_mel = input_features[i]
            prediction = model.predict(input_mel)
            label = labels[i].tolist()
            print(f"File: {file_names[i]}")
            print("Predicted Token IDs:", prediction)
            print("Target Token IDs:", label)

        break  # Remove or adjust this as needed


if __name__ == "__main__":
    main()
