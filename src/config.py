import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model Settings
# Using Phi-3 Mini (3.8B) - Local GGUF
MODEL_NAME = "Phi-3-mini-4k-instruct-q4.gguf" 
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# Embedding Model (CPU friendly)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB Settings
COLLECTION_NAME = "research_papers"

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OCR Settings
# Ensure Tesseract is installed on the system
# sudo apt-get install tesseract-ocr (Linux)
# brew install tesseract (Mac)
