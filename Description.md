# Description.md

## Technical Features
- **Model**: NVIDIA NeMo `stt_hi_conformer_ctc_medium` (Hindi, CTC, BPE)
- **ONNX Export**: Model exported using NeMo toolkit, inference via ONNXRuntime
- **Audio Preprocessing**: Dither, pre-emphasis, log-Mel spectrograms (80 bins, librosa)
- **FastAPI**: `/transcribe` endpoint, async handling, validation for file type/duration
- **Test Suite**: Scripted tests for API and model

## Known Issues & Limitations
- Only short audio (5-10s, 16kHz mono WAV) is supported by default
- Blank token index is assumed to be last (verify with vocabulary)
- No language detection (Hindi-only)
- No confidence scores in output (future work)
- No automatic segmentation for long audio (future work)

## Mitigations
- Input validation prevents unsupported files
- CTC decoding uses vocabulary and blank index as per export script
- Docker ensures reproducibility and dependency isolation

## Assumptions
- Input audio is clean speech, minimal background noise
- Model/vocabulary are not corrupted
- System has enough RAM for ONNX model inference

## Performance Notes
- Inference latency: depends on hardware, typically < 1s per 5-10s clip (CPU)
- Preprocessing pipeline is vectorized for efficiency
- Docker image is multi-stage and slim for fast deployment

## Optimization Opportunities
- Batch processing for multiple files
- Caching for repeated requests
- Use GPU ONNXRuntime for faster inference (if available)
- Profile and optimize Mel spectrogram pipeline

## References
- [NVIDIA NeMo ASR Models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
