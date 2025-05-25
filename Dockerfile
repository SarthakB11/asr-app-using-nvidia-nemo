# Stage 1: Base image with dependencies
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM base AS final

WORKDIR /app

# Copy application code and scripts
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p ./models ./downloads

# Copy the NeMo model file and pre-converted ONNX model
COPY downloads/stt_hi_conformer_ctc_medium.nemo ./downloads/
COPY models/stt_hi_conformer_ctc_medium.onnx ./models/
COPY models/vocabulary.json ./models/

# Create startup script file
COPY <<EOF /app/start.sh
#!/bin/bash

# Check if ONNX model already exists
if [ ! -f ./models/stt_hi_conformer_ctc_medium.onnx ] || [ ! -f ./models/vocabulary.json ]; then
  # Check if NeMo model file exists
  if [ ! -f ./downloads/stt_hi_conformer_ctc_medium.nemo ]; then
    echo "ERROR: NeMo model file not found at ./downloads/stt_hi_conformer_ctc_medium.nemo"
    echo "Please ensure the model file is copied to the container."
    exit 1
  fi

  # Convert model to ONNX
  echo "ONNX model or vocabulary not found. Converting NeMo model to ONNX format..."
  python scripts/convert_to_onnx.py

  # Check if ONNX model was created successfully
  if [ ! -f ./models/stt_hi_conformer_ctc_medium.onnx ]; then
    echo "ERROR: ONNX model conversion failed."
    exit 1
  fi
else
  echo "ONNX model and vocabulary already exist. Skipping conversion."
fi

# Start the application
echo "Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
EOF

# Make startup script executable
RUN chmod +x /app/start.sh

# Expose the port the app runs on
EXPOSE 8000

# Add /app to PYTHONPATH to ensure modules are found
ENV PYTHONPATH=/app

# Command to run the startup script
CMD ["/app/start.sh"]
