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

## Features
- Hindi ASR using NVIDIA NeMo model (ONNX)
- REST API (`/transcribe`) for audio file transcription
- Audio preprocessing: dither, pre-emphasis, log-Mel spectrograms (80 bins)
- Dockerized for reproducibility
- Test scripts and sample usage

## Installation & Setup
### Local (Python v3.10 recommended)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Download Model
Place the NeMo model in `downloads/` (already present if following plan):
```
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium
```

Convert to ONNX (if not already):
```bash
venv/bin/python scripts/convert_to_onnx.py
```

### Run Locally
```bash
venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
Build and run:
```bash
docker build -t hindi-asr-app .
docker run -p 8000:8000 hindi-asr-app
```

## API Usage
### Transcribe Endpoint
- **POST** `/transcribe`
- **Form-data**: `file` (audio/wav, 16kHz, mono, 5-10s)

#### Example (curl)
```bash
curl -X POST "http://localhost:8000/transcribe" -F "file=@temp_audio/hindi_sample.wav"
```

#### Example (Python)
```python
import requests
with open('temp_audio/hindi_sample.wav', 'rb') as f:
    r = requests.post('http://localhost:8000/transcribe', files={'file': f})
    print(r.json())
```

## Testing
Run the provided test script:
```bash
venv/bin/python scripts/test_transcription.py --num-files 3
```

## Troubleshooting
- **503 Service Unavailable**: Model failed to load. Ensure ONNX file is valid and present in `models/`.
- **Address already in use**: Free port 8000 before starting server.
- **Audio not transcribing**: Check that input is WAV, 16kHz, mono, and 5-10s duration.

## Deployment
- See `DEPLOYMENT.md` for cloud/edge setup, CI/CD, and monitoring.

---
For technical details, see `Description.md`.
