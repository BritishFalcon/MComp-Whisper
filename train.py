from model_utils import *

def main():
    # Paths
    train_audio_dir = "synthetic_data/train"
    train_midi_dir = "synthetic_data/train"
    val_audio_dir = "synthetic_data/test"
    val_midi_dir = "synthetic_data/test"
    checkpoint_path = "checkpoints/model_epoch_5.pt"  # Update as needed

    # Initialize tokenizer and feature extractor
    tokenizer = initialize_tokenizer()
    feature_extractor = initialize_feature_extractor()

    # Initialize model
    model = WhisperREMIModel(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        checkpoint_path=checkpoint_path
    )

    # Create data loaders
    train_loader = create_data_loader(
        audio_dir=train_audio_dir,
        midi_dir=train_midi_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=8
    )
    val_loader = create_validation_loader(
        audio_dir=val_audio_dir,
        midi_dir=val_midi_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=4
    )

    # Optimizer
    optimizer = model.get_optimizer(lr=1e-4)

    # Train with validation
    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=5,
        log_file="training_log.txt"
    )

if __name__ == "__main__":
    main()

