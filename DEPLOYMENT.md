# Deployment Guide

## Local Deployment
See `README.md` for local and Docker instructions.

## Cloud Deployment (Example: Azure, AWS, GCP)
1. Build Docker image:
   ```bash
   docker build -t hindi-asr-app .
   ```
2. Push to container registry (e.g., Docker Hub, AWS ECR, GCP Artifact Registry)
3. Deploy using your cloud provider's container service (e.g., Azure App Service, AWS ECS, GCP Cloud Run)
4. Ensure port 8000 is exposed and model files are present in `/app/models/`

## Edge Deployment
- Use Docker on edge device (Jetson, NUC, etc.)
- Optimize image (multi-stage, minimal dependencies)
- Mount model files as volumes if device storage is limited

## CI/CD Pipeline
- Example: GitHub Actions (`.github/workflows/ci.yml`)
- Lint, test, and optionally build Docker image on push/PR

## Monitoring & Logging
- FastAPI logs to stdout (capture with cloud/edge logging)
- Add Prometheus/Grafana for metrics (future work)
- Monitor API latency and error rates

## Maintenance & Update Strategy
- Regularly rebuild Docker image to update dependencies
- Monitor NVIDIA NeMo for new model releases
- Update ONNX export if NeMo model is updated
- Back up vocabulary and model files
