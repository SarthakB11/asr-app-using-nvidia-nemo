
# FastAPI ASR Application (Hindi, ONNX, NVIDIA NeMo)

## Overview
This project provides a production-ready Automatic Speech Recognition (ASR) REST API for Hindi, using a FastAPI backend and an ONNX-optimized NVIDIA NeMo `stt_hi_conformer_ctc_medium` model. It supports transcription of short audio clips (5-10 seconds, 16kHz WAV) and is containerized for easy deployment.

## Architecture
```
+--------+      +----------------+      +---------------------+
| Client | ---> | FastAPI Server | -->  | ONNXRuntime Inference|
+--------+      +----------------+      +---------------------+
                                  |     (Hindi Conformer CTC)
                                  |
                        [Audio Preprocessing (librosa, numpy)]
```

## Project Video
[screen-capture.webm](https://github.com/user-attachments/assets/9def8a09-7a2b-4711-937b-fe8a7ef44d1c)

## Key Components

### 1. FastAPI Application (`app/main.py`)
- REST API endpoints for audio transcription
- Asynchronous request handling
- Input validation and error handling
- CORS middleware for web compatibility

### 2. Audio Preprocessing (`app/audio_utils.py`)
- Audio loading and validation (16kHz, mono WAV)
- Signal processing pipeline:
  - Dithering for numerical stability
  - Pre-emphasis filter
  - Log-Mel spectrogram extraction (80 bins)
  - Audio duration validation

### 3. ASR Inference (`app/asr_inference.py`)
- ONNX model loading and inference
- CTC decoding
- Batch processing support
- Model input/output handling

### 4. Model Conversion (`scripts/convert_to_onnx.py`)
- Downloads the `stt_hi_conformer_ctc_medium` model from NVIDIA NeMo
- Converts the model to ONNX format
- Saves the model with proper node names for inference

### 5. Docker Configuration (`Dockerfile`)
- Multi-stage build for optimized image size
- Python 3.10 base image
- Dependency installation
- Non-root user for security
- Health check endpoint

## Screenshots
![Screenshot from 2025-05-25 02-12-07](https://github.com/user-attachments/assets/98c1b7ee-b638-4817-9862-2f4651ff8302)
![Screenshot from 2025-05-25 02-12-14](https://github.com/user-attachments/assets/e56dcabc-a6b4-4e83-ab73-7bfb70123f4e)

## Features
- **High Accuracy**: Based on NVIDIA's Conformer CTC model trained on Hindi speech
- **Low Latency**: ONNX Runtime optimized inference
- **Scalable**: Containerized deployment ready
- **Developer Friendly**: Well-documented API and code
- **Production Ready**: Health checks, logging, and error handling

## Installation & Setup

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- NVIDIA GPU (recommended) or CPU

### 1. Local Development Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Model Setup
#### Option A: Download Pre-converted ONNX Model
```bash
mkdir -p models
# Download stt_hi_conformer_ctc_medium.onnx to models/
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium
```

#### Option B: Convert from NeMo Model
1. Download the NeMo model:
   ```bash
   mkdir -p downloads
   # Download stt_hi_conformer_ctc_medium.nemo to downloads/
   ```

2. Convert to ONNX:
   ```bash
   python scripts/convert_to_onnx.py \
     --nemo-path downloads/stt_hi_conformer_ctc_medium.nemo \
     --output-dir models
   ```

### 3. Run the Application
#### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode (with Gunicorn)
```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000 --workers 4
```

### 4. Docker Deployment
```bash
# Build the image
docker build -t hindi-asr-app .

# Run the container
docker run -d \
  --name asr-app \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  hindi-asr-app
```

## API Documentation

### 1. Transcribe Audio
- **Endpoint**: `POST /transcribe`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Audio file (WAV, 16kHz, mono, 5-10s)
- **Response**:
  ```json
  {
    "status": "success",
    "text": "transcribed text in Hindi",
    "duration": 5.43,
    "language": "hi"
  }
  ```

### 2. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
  }
  ```

## Example Usage

### cURL
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -F "file=@temp_audio/hindi_sample.wav"
```

### Python Client
```python
import requests

def transcribe_audio(file_path, server_url="http://localhost:8000"):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{server_url}/transcribe", files=files)
        return response.json()

# Example usage
result = transcribe_audio("temp_audio/hindi_sample.wav")
print(f"Transcribed Text: {result['text']}")
print(f"Processing Time: {result['duration']:.2f} seconds")
```

## Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Test
```bash
python scripts/test_transcription.py \
  --input-dir sample_audio \
  --output results.json \
  --num-files 5
```

## Troubleshooting
- **503 Service Unavailable**: Model failed to load. Ensure ONNX file is valid and present in `models/`.
- **Address already in use**: Free port 8000 before starting server.
- **Audio not transcribing**: Check that input is WAV, 16kHz, mono, and 5-10s duration.

## Deployment
- See `DEPLOYMENT.md` for cloud/edge setup, CI/CD, and monitoring.

---
For technical details, see `Description.md`.
