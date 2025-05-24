import soundfile as sf
import numpy as np
from fastapi import UploadFile, HTTPException, status
import io
from pydub import AudioSegment
import logging

# Import configuration
from .config import TARGET_SAMPLE_RATE, MIN_DURATION_S, MAX_DURATION_S

# Configure logger for this module
logger = logging.getLogger(__name__)

async def validate_and_load_audio(file: UploadFile) -> np.ndarray:
    """
    Validates audio file properties (WAV, TARGET_SAMPLE_RATE, duration limits) 
    and loads it as a mono float32 numpy array.
    """
    logger.info(f"Validating audio file: {file.filename}, content type: {file.content_type}")

    if not (file.content_type == "audio/wav" or file.filename.lower().endswith(".wav")):
        logger.warning(f"Invalid file type for {file.filename}. Expected WAV, got {file.content_type}.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only WAV audio is accepted."
        )

    try:
        audio_bytes = await file.read()
        # Reset file pointer if other parts of the app might have read it or need to re-read.
        # For pydub, it's good practice to ensure it gets the full stream.
        await file.seek(0) 
        
        # Use pydub to check duration and perform initial conversions (mono, sample rate)
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        except Exception as e_pydub:
            logger.error(f"Pydub could not process file {file.filename}: {e_pydub}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not process audio file. Ensure it is a valid audio format. Error: {e_pydub}"
            )

        duration_s = len(audio_segment) / 1000.0

        logger.info(f"Audio file: {file.filename}, Original duration: {duration_s:.2f}s, Channels: {audio_segment.channels}, Frame Rate: {audio_segment.frame_rate}")

        if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
            logger.warning(f"Audio duration for {file.filename} ({duration_s:.2f}s) is out of range [{MIN_DURATION_S}-{MAX_DURATION_S}s].")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio duration must be between {MIN_DURATION_S} and {MAX_DURATION_S} seconds. Received {duration_s:.2f}s."
            )

        # Convert to mono if necessary
        if audio_segment.channels != 1:
            logger.info(f"Converting {file.filename} to mono.")
            audio_segment = audio_segment.set_channels(1)
        
        # Resample to target sample rate if necessary
        if audio_segment.frame_rate != TARGET_SAMPLE_RATE:
            logger.info(f"Resampling {file.filename} to {TARGET_SAMPLE_RATE}Hz from {audio_segment.frame_rate}Hz.")
            audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE)

        # Export to WAV format in-memory to pass to soundfile
        # This ensures soundfile gets a clean, standardized WAV stream
        processed_audio_stream = io.BytesIO()
        audio_segment.export(processed_audio_stream, format="wav")
        processed_audio_stream.seek(0)

        # Use soundfile to read the processed audio data (now guaranteed mono, TARGET_SAMPLE_RATE WAV)
        with sf.SoundFile(processed_audio_stream, 'r') as audio_sf_file:
            # Double-check properties after pydub processing, though they should be correct
            if audio_sf_file.samplerate != TARGET_SAMPLE_RATE:
                logger.error(f"Soundfile check failed: {file.filename} sample rate is {audio_sf_file.samplerate} after pydub, expected {TARGET_SAMPLE_RATE}.")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail="Internal error: Audio sample rate mismatch after processing."
                )
            if audio_sf_file.channels != 1:
                logger.error(f"Soundfile check failed: {file.filename} channels is {audio_sf_file.channels} after pydub, expected 1.")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal error: Audio channel mismatch after processing."
                )
            
            audio_data = audio_sf_file.read(dtype='float32') # Read as float32 for NeMo models
            
            # --- Peak Normalization Step ---
            logger.info(f"Audio_data before peak normalization: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}")
            peak = np.max(np.abs(audio_data))
            if peak > 1e-5: # Avoid division by zero or near-zero for silent audio
                audio_data = audio_data / peak
            logger.info(f"Audio_data after peak normalization: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}")
            # --- End Peak Normalization Step ---

            # --- Pre-emphasis Filter ---
            preemphasis_coeff = 0.97
            audio_data = np.append(audio_data[0], audio_data[1:] - preemphasis_coeff * audio_data[:-1])
            logger.info(f"Audio_data after pre-emphasis: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}")
            # --- End Pre-emphasis Filter ---

        logger.info(f"Successfully validated and loaded audio from {file.filename}. Shape: {audio_data.shape}")
        return audio_data

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except sf.SoundFileError as e_sf:
        logger.error(f"Soundfile could not read audio file {file.filename}: {e_sf}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read audio file. Ensure it is a valid WAV format. Soundfile error: {e_sf}"
        )
    except Exception as e:
        logger.error(f"Unexpected error processing audio file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the audio file: {str(e)}"
        )
