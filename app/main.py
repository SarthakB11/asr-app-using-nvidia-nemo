import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pythonjsonlogger import jsonlogger

# Import configurations and utility functions
from .config import LOG_LEVEL, ONNX_MODEL_PATH, VOCAB_FILE_PATH
from .audio_utils import validate_and_load_audio
from .asr_inference import load_model, transcribe_audio

# --- Logger Setup ---
logger = logging.getLogger("asr_app")  # Use a specific name for the app logger
logger.setLevel(LOG_LEVEL)

log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s "
        "%(lineno)d %(message)s"
)
log_handler.setFormatter(formatter)

logger.addHandler(log_handler)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="NVIDIA NeMo ASR API",
    description=(
        "An API for transcribing audio using an ONNX model exported from "
        "NVIDIA NeMo."
    ),
    version="1.0.0"
)

# Set up static files directory
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

# Create directories if they don't exist
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(templates_dir))

# --- Application State ---
app_state = {"is_model_loaded": False}

# --- Application Lifecycle Events ---


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing...")
    # Initial check for model and vocab files for early feedback
    model_exists = os.path.exists(ONNX_MODEL_PATH)
    vocab_exists = os.path.exists(VOCAB_FILE_PATH)

    if not model_exists:
        logger.error(
            f"ONNX model file not found at path: {ONNX_MODEL_PATH}. "
            "Transcription will fail."
        )
    if not vocab_exists:
        logger.error(
            f"Vocabulary file not found at path: {VOCAB_FILE_PATH}. "
            "Transcription will fail."
        )

    if not model_exists or not vocab_exists:
        app_state["is_model_loaded"] = False
    else:
        logger.info("ONNX model and vocabulary files found.")
        logger.info("Attempting to load model and vocabulary.")
        try:
            # Pass both paths
            load_model(model_path=ONNX_MODEL_PATH, vocab_path=VOCAB_FILE_PATH)
            app_state["is_model_loaded"] = True
            logger.info("ASR model and vocabulary loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load ASR model or vocabulary: {e}",
                exc_info=True
            )
            app_state["is_model_loaded"] = False
    logger.info("Application startup complete.")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown: Cleaning up resources...")
    # Add any cleanup logic here if needed
    logger.info("Application shutdown complete.")


# --- API Endpoints ---


@app.get("/health", summary="Health Check", tags=["General"])
async def health_check():
    logger.info("Health check endpoint called.")
    return JSONResponse(
        content={
            "status": "healthy",
            "model_loaded": app_state.get("is_model_loaded", False)
        },
        status_code=status.HTTP_200_OK
    )


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """Serve the main application page."""
    logger.info("Main page requested")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe", summary="Transcribe Audio File", tags=["ASR"])
async def transcribe_endpoint(file: UploadFile = File(...)):
    logger.info(
        f"Received file for transcription: {file.filename}, "
        f"content_type: {file.content_type}"
    )

    if not app_state.get("is_model_loaded", False):
        logger.error(
            "Transcription attempt failed: ASR model is not loaded."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR model is not available. Service is unavailable."
        )

    try:
        logger.info(
            f"TRANS_ENDPOINT: Validating audio for {file.filename}"
        )
        log_mel_spectrogram, num_frames = await validate_and_load_audio(file)
        logger.info(
            f"TRANS_ENDPOINT: Audio validated. "
            f"Shape: {log_mel_spectrogram.shape}, Frames: {num_frames}"
        )

        logger.info(
            f"TRANS_ENDPOINT: Transcribing spectrogram for {file.filename}"
        )
        transcription = await transcribe_audio(log_mel_spectrogram, num_frames)
        logger.info(
            f"TRANS_ENDPOINT: Transcription successful: {transcription}"
        )

        # Return response
        return JSONResponse(
            content={
                "filename": file.filename,
                "transcription": transcription
            }
        )

    except HTTPException as e:
        # Logged in audio_utils or here if it's from this endpoint's logic
        logger.warning(
            f"TRANS_ENDPOINT: HTTPException for {file.filename}: "
            f"{e.detail}",
            exc_info=True
        )
        raise e  # Re-raise HTTPException to be handled by FastAPI
    except Exception as e:
        logger.error(
            f"TRANS_ENDPOINT: Unexpected error for {file.filename}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # This block is for local development/debugging.
    # For production, use uvicorn app.main:app --host 0.0.0.0 --port 8000
    logger.info(
        "Starting Uvicorn server for local development on http://0.0.0.0:8000"
    )
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
