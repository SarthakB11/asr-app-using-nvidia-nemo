import os

# Model Configuration
ONNX_MODEL_NAME = os.getenv(
    "ONNX_MODEL_NAME", "stt_hi_conformer_ctc_medium.onnx"
)
# Points to project_root/models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, ONNX_MODEL_NAME)

VOCAB_FILENAME = os.getenv("VOCAB_FILENAME", "vocabulary.json")
VOCAB_FILE_PATH = os.path.join(MODELS_DIR, VOCAB_FILENAME)

# Audio Processing Configuration
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))  # Hz
MIN_DURATION_S = float(os.getenv("MIN_DURATION_S", "1.0"))  # seconds
MAX_DURATION_S = float(os.getenv("MAX_DURATION_S", "30.0"))  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
