import os
import shutil
import random

# Paths
source_folder = "Synthesized dataset"
train_folder = "train"
test_folder = "test"

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Collect all unique filenames
all_files = [os.path.splitext(f)[0] for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
unique_files = list(set(all_files))  # Ensure unique

# Shuffle the files list for randomness
random.shuffle(unique_files)

# Split into train and test sets (80/20)
split_index = int(len(unique_files) * 0.8)
train_files = unique_files[:split_index]
test_files = unique_files[split_index:]

# Helper function to move files
def move_files(file_list, destination_folder):
    for base_name in file_list:
        for ext in ['.mid', '.wav']:
            file_name = base_name + ext
            source_path = os.path.join(source_folder, file_name)
            dest_path = os.path.join(destination_folder, file_name)
            if os.path.exists(source_path):
                shutil.move(source_path, dest_path)

# Move train files
move_files(train_files, train_folder)

# Move test files
move_files(test_files, test_folder)

print(f"Files successfully split and moved:")
print(f"Train: {len(train_files)}")
print(f"Test: {len(test_files)}")
