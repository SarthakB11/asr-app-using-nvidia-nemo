# scripts/download_dataset.py
import kagglehub
import os

# Define the path for kaggle.json within the project structure if needed
# However, kagglehub typically uses ~/.kaggle/kaggle.json by default

# Ensure the KAGGLE_CONFIG_DIR is set if you want to use a non-default location
# For this session, we've already copied it to ~/.kaggle/kaggle.json

print("Starting dataset download...")
try:
    # Download latest version of the dataset
    # The path returned is where the dataset files are stored
    path = kagglehub.dataset_download(
        "hmsolanki/indian-languages-audio-dataset"
    )
    print("Dataset download complete.")
    print(f"Path to dataset files: {path}")

    # Let's list the contents of the downloaded path to verify
    if path and os.path.exists(path):
        print("Contents of the dataset directory:")
        for item in os.listdir(path):
            print(f"  - {item}")
    else:
        print("Dataset path does not exist or was not returned.")

except Exception as e:
    print(f"An error occurred during dataset download: {e}")
    print(
        "Please ensure your Kaggle API key (kaggle.json) is correctly set up "
        "at ~/.kaggle/kaggle.json"
    )
