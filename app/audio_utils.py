import soundfile as sf
import numpy as np
from fastapi import UploadFile, HTTPException, status
import io
from pydub import AudioSegment
import logging
import librosa

# Import configuration
from .config import TARGET_SAMPLE_RATE, MIN_DURATION_S, MAX_DURATION_S

# Configure logger for this module
logger = logging.getLogger(__name__)

# Spectrogram parameters (typical for Conformer)
N_MELS = 80
N_FFT = 400  # For 25ms window at 16kHz (16000 * 0.025)
HOP_LENGTH = 160  # For 10ms hop at 16kHz (16000 * 0.01)
DITHER_COEFF = 1e-5
PREEMPHASIS_COEFF = 0.97

async def validate_and_load_audio(file: UploadFile) -> tuple[np.ndarray, int]:
    """
    Validates audio file properties, applies preprocessing (dither, preemphasis),
    and converts it to a log-Mel spectrogram.
    Returns:
        tuple[np.ndarray, int]: A tuple containing:
            - log_mel_spectrogram (np.ndarray): The log-Mel spectrogram with shape (N_MELS, num_frames).
            - num_frames (int): The number of frames in the spectrogram.
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
        await file.seek(0)
        
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

        if audio_segment.channels != 1:
            logger.info(f"Converting {file.filename} to mono.")
            audio_segment = audio_segment.set_channels(1)
        
        if audio_segment.frame_rate != TARGET_SAMPLE_RATE:
            logger.info(f"Resampling {file.filename} to {TARGET_SAMPLE_RATE}Hz from {audio_segment.frame_rate}Hz.")
            audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE)

        processed_audio_stream = io.BytesIO()
        audio_segment.export(processed_audio_stream, format="wav")
        processed_audio_stream.seek(0)

        with sf.SoundFile(processed_audio_stream, 'r') as audio_sf_file:
            if audio_sf_file.samplerate != TARGET_SAMPLE_RATE:
                logger.error(f"Soundfile check failed: {file.filename} sample rate is {audio_sf_file.samplerate} after pydub, expected {TARGET_SAMPLE_RATE}.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error: Audio sample rate mismatch after processing.")
            if audio_sf_file.channels != 1:
                logger.error(f"Soundfile check failed: {file.filename} channels is {audio_sf_file.channels} after pydub, expected 1.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error: Audio channel mismatch after processing.")
            
            audio_data = audio_sf_file.read(dtype='float32')
            logger.info(f"Audio_data after soundfile.read(dtype='float32'): min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}")

            # --- Dithering Step ---
            audio_data_dithered = audio_data + DITHER_COEFF * np.random.randn(*audio_data.shape)
            audio_data_dithered = np.clip(audio_data_dithered, -1.0, 1.0) # Clip after dither
            logger.info(f"Audio_data after dithering: min={np.min(audio_data_dithered):.4f}, max={np.max(audio_data_dithered):.4f}, mean={np.mean(audio_data_dithered):.4f}")

            # --- Pre-emphasis Filter ---
            audio_data_preemphasized = np.append(audio_data_dithered[0], audio_data_dithered[1:] - PREEMPHASIS_COEFF * audio_data_dithered[:-1])
            logger.info(f"Audio_data after pre-emphasis: min={np.min(audio_data_preemphasized):.4f}, max={np.max(audio_data_preemphasized):.4f}, mean={np.mean(audio_data_preemphasized):.4f}")

            # --- Mel Spectrogram Calculation ---
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data_preemphasized,
                sr=TARGET_SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                window='hann' # Common window function
            )

            # --- Convert to Log-Mel Spectrogram (dB scale) ---
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # --- Per-feature Mean/Variance Normalization ---
            if log_mel_spectrogram.size > 0: 
                mean = np.mean(log_mel_spectrogram, axis=1, keepdims=True)
                std_dev = np.std(log_mel_spectrogram, axis=1, keepdims=True)
                log_mel_spectrogram = (log_mel_spectrogram - mean) / (std_dev + 1e-5) 
                logger.info(f"Log-Mel spectrogram AFTER per-feature normalization: "
                            f"min={np.min(log_mel_spectrogram):.4f}, max={np.max(log_mel_spectrogram):.4f}, "
                            f"mean={np.mean(log_mel_spectrogram):.4f}, std={np.std(log_mel_spectrogram):.4f}")
            else:
                logger.warning("Log-Mel spectrogram is empty before per-feature normalization.")
            num_frames = log_mel_spectrogram.shape[1]

        logger.info(f"Successfully generated log-Mel spectrogram from {file.filename}. Shape: {log_mel_spectrogram.shape}")
        return log_mel_spectrogram, num_frames

    except HTTPException:
        raise
    except sf.SoundFileError as e_sf:
        logger.error(f"Soundfile could not read audio file {file.filename}: {e_sf}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read audio file. Ensure it is a valid WAV format. Soundfile error: {e_sf}")
    except Exception as e:
        logger.error(f"Unexpected error processing audio file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred while processing the audio file: {str(e)}")
