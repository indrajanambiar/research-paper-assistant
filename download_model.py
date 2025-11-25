from huggingface_hub import hf_hub_download
import os

# Configuration
# Phi-3 Mini (3.8B) - Non-gated, runs on CPU
REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
MODELS_DIR = "models"

def download_model():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print(f"Downloading {FILENAME} from {REPO_ID}...")
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please try downloading manually from HuggingFace.")

if __name__ == "__main__":
    download_model()
