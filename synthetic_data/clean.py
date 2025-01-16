import os

def clean_unpaired_files(folder_path):

    all_files = os.listdir(folder_path)

    # Separate files into .wav and .mid sets
    wav_files = {os.path.splitext(file)[0] for file in all_files if file.endswith('.wav')}
    mid_files = {os.path.splitext(file)[0] for file in all_files if file.endswith('.mid')}

    # Identify unpaired files
    unpaired_wav = wav_files - mid_files
    unpaired_mid = mid_files - wav_files

    # Delete unpaired files
    for unpaired in unpaired_wav:
        wav_path = os.path.join(folder_path, f"{unpaired}.wav")
        os.remove(wav_path)
        print(f"Removed unpaired .wav file: {wav_path}")

    for unpaired in unpaired_mid:
        mid_path = os.path.join(folder_path, f"{unpaired}.mid")
        os.remove(mid_path)
        print(f"Removed unpaired .mid file: {mid_path}")

    print("Cleanup complete.")

if __name__ == "__main__":
    folder = "train"
    clean_unpaired_files(folder)
