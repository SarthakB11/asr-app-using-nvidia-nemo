# Deployment Guide

## Local Deployment

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t hindi-asr-app .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --name hindi-asr-app-container hindi-asr-app
   ```

3. The container will automatically:
   - Download the NeMo model if not present
   - Convert the model to ONNX format
   - Start the FastAPI application on port 8000

4. Access the application at http://localhost:8000

### Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the NeMo model:
   ```bash
   mkdir -p downloads
   wget -q https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium/files/stt_hi_conformer_ctc_medium.nemo -O ./downloads/stt_hi_conformer_ctc_medium.nemo
   ```

3. Convert the model to ONNX format:
   ```bash
   python scripts/convert_to_onnx.py
   ```

4. Start the FastAPI application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Cloud Deployment

### Container-Based Deployment (Azure, AWS, GCP)

1. Build Docker image:
   ```bash
   docker build -t hindi-asr-app .
   ```

2. Push to container registry (e.g., Docker Hub, AWS ECR, GCP Artifact Registry)

3. Deploy using your cloud provider's container service:
   - Azure: Azure Container Instances or App Service
   - AWS: ECS, EKS, or App Runner
   - GCP: Cloud Run or GKE

4. The container will automatically handle model download and conversion

## Edge Deployment
- Use Docker on edge device (Jetson, NUC, etc.)
- Optimize image (multi-stage, minimal dependencies)
- Mount model files as volumes if device storage is limited

## CI/CD Pipeline

### GitHub Actions

The project includes a CI/CD pipeline in `.github/workflows/ci.yml` that:

1. **Build and Test:**
   - Checks out the code with Git LFS support
   - Sets up Python environment
   - Installs dependencies
   - Runs linting with flake8
   - Downloads and converts the NeMo model
   - Starts the FastAPI application
   - Runs tests
   - Builds the Docker image

2. **Deployment to GitHub Pages:**
   - Creates a static website with:
     - HTML, CSS, and JavaScript for the frontend
     - API documentation
     - Model files
   - Deploys to GitHub Pages

### Customizing the CI/CD Pipeline

To modify the CI/CD pipeline:

1. Edit `.github/workflows/ci.yml`
2. Add additional steps as needed (e.g., security scanning, additional tests)
3. Configure environment variables for deployment

### GitHub Pages Deployment

The GitHub Pages deployment creates a static website that includes:

- A responsive frontend interface
- API documentation
- Model files for client-side processing

## Monitoring & Logging
- FastAPI logs to stdout (capture with cloud/edge logging)
- Add Prometheus/Grafana for metrics (future work)
- Monitor API latency and error rates

## Maintenance & Update Strategy

### Model Updates

- Monitor NVIDIA NeMo for new model releases
- When a new model is released:
  1. Update the model URL in `scripts/convert_to_onnx.py`
  2. Update the model filename if necessary
  3. Test the conversion process locally
  4. Update the Dockerfile and CI workflow if needed

### Dependency Management

- Regularly rebuild Docker image to update dependencies
- Update `requirements.txt` with specific versions
- Test compatibility with new versions of key dependencies:
  - nemo_toolkit
  - onnx
  - onnxruntime
  - fastapi

### Backup Strategy

- Use Git LFS to track model files
- Back up vocabulary and model files to secure storage
- Document model versions and conversion parameters

### Monitoring and Maintenance

- Schedule regular health checks
- Monitor API performance and error rates
- Update documentation as the application evolves
