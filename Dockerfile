# Stage 1: Builder stage - To install dependencies
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /opt/app

# Install system dependencies
# libsndfile1 is for librosa/soundfile, ffmpeg for pydub
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce layer size
# Target a specific directory for easy copying to the next stage
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -t /opt/app/packages

# Stage 2: Final stage - To create the lean application image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required at runtime (if any, beyond what's in slim)
# For now, assuming libsndfile1 and ffmpeg are also needed at runtime by the libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /opt/app/packages /usr/local/lib/python3.10/site-packages

# Copy application code
COPY ./app ./app

# Copy models - Assuming models are in a 'models' directory at the root of the project
# and app/config.py or similar points to /app/models/
COPY ./models ./models

# Expose the port the app runs on
EXPOSE 8000

# Add /app to PYTHONPATH to ensure modules are found
ENV PYTHONPATH=/app

# Command to run the application
# Ensure app.main:app is the correct path to your FastAPI app instance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
