import os
import pandas as pd
from huggingface_hub import create_repo, upload_file

# Get Hugging Face token from environment variables (GitHub Actions secret)
HF_TOKEN = os.environ.get('HF_TOKEN')

# Define the path to your dataset
DATA_FILE_PATH = 'tourism_project/data/tourism.csv'

# Define your Hugging Face username and the dataset repo name
# IMPORTANT: Replace 'your-hf-username' with your actual Hugging Face username
HF_USERNAME = 'your-hf-username'
HF_DATASET_REPO = f'{HF_USERNAME}/tourism-dataset'

def register_dataset_on_hf():
    if not HF_TOKEN:
        print("HF_TOKEN environment variable not set. Cannot upload dataset.")
        return

    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Dataset file not found at {DATA_FILE_PATH}")
        return

    try:
        print(f"Creating/checking Hugging Face dataset repo: {HF_DATASET_REPO}")
        create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

        print(f"Uploading {DATA_FILE_PATH} to {HF_DATASET_REPO}")
        upload_file(
            path_or_fileobj=DATA_FILE_PATH,
            path_in_repo='tourism.csv',
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("Dataset uploaded successfully!")

    except Exception as e:
        print(f"An error occurred during dataset registration: {e}")

if __name__ == '__main__':
    register_dataset_on_hf()
