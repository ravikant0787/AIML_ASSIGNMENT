import os
import shutil
from huggingface_hub import create_repo, upload_folder

# Get Hugging Face token from environment variables (GitHub Actions secret)
HF_TOKEN = os.environ.get('HF_TOKEN')

# Define paths for deployment files and MLflow artifacts
DEPLOYMENT_APP_FILE = 'tourism_project/deployment/app.py'
DEPLOYMENT_DOCKERFILE = 'tourism_project/deployment/Dockerfile'
DEPLOYMENT_REQUIREMENTS_FILE = 'tourism_project/deployment/requirements.txt'
MLRUNS_DIR = 'tourism_project/mlruns'

# Define your Hugging Face username and the Space repo name
# IMPORTANT: Replace 'your-hf-username' with your actual Hugging Face username
HF_USERNAME = 'your-hf-username'
HF_SPACE_REPO = f'{HF_USERNAME}/tourism-prediction-space'

def deploy_to_hf_space():
    if not HF_TOKEN:
        print("HF_TOKEN environment variable not set. Cannot deploy to Space.")
        return

    # Create a temporary directory to stage files for upload
    temp_deploy_dir = 'temp_hf_deploy'
    os.makedirs(temp_deploy_dir, exist_ok=True)

    try:
        # Copy necessary files to the temporary directory
        shutil.copy(DEPLOYMENT_APP_FILE, temp_deploy_dir)
        shutil.copy(DEPLOYMENT_DOCKERFILE, temp_deploy_dir)
        shutil.copy(DEPLOYMENT_REQUIREMENTS_FILE, temp_deploy_dir)

        # Copy the entire mlruns directory
        if os.path.exists(MLRUNS_DIR):
            shutil.copytree(MLRUNS_DIR, os.path.join(temp_deploy_dir, 'mlruns'), dirs_exist_ok=True)
        else:
            print(f"Warning: {MLRUNS_DIR} not found. Model artifacts will be missing.")

        print(f"Creating/checking Hugging Face Space repo: {HF_SPACE_REPO}")
        create_repo(repo_id=HF_SPACE_REPO, repo_type="space", token=HF_TOKEN, exist_ok=True)

        print(f"Uploading content of {temp_deploy_dir} to {HF_SPACE_REPO}")
        upload_folder(
            folder_path=temp_deploy_dir,
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            token=HF_TOKEN,
            commit_message="Update deployment files from GitHub Actions"
        )
        print("Deployment files uploaded to Hugging Face Space successfully!")

    except Exception as e:
        print(f"An error occurred during deployment to Hugging Face Space: {e}")
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_deploy_dir):
            shutil.rmtree(temp_deploy_dir)

if __name__ == '__main__':
    deploy_to_hf_space()
