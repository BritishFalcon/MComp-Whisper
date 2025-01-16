from model_utils import *

# DONE: Run with the new startwith method of freezing
# DONE: Run with decoder-only training
# DONE: Explore following decoder-only
# DONE: Run on polyphonic data


def main():
    
    train_audio_dir = "real_data/train"
    train_midi_dir = "real_data/train"
    val_audio_dir = "real_data/test"
    val_midi_dir = "real_data/test"
    checkpoint_path = None  # Update as needed

    tokenizer = initialize_tokenizer()
    feature_extractor = initialize_feature_extractor()

    model = WhisperREMIModel(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        checkpoint_path=checkpoint_path,
    )

    # Data loading
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
        batch_size=8
    )

    # Domain standard but still low learning rate for best chance of convergence
    optimizer = model.get_optimizer(lr=1e-4)

    # Train with validation (something like 1e4 or crazy to prevent eval)
    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=100,
        log_file="training_log.txt",
        eval_every=1,
    )


if __name__ == "__main__":
    main()
