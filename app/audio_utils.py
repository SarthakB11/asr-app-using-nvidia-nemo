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
    """Validates audio file properties and creates spectrogram.
    Returns:
        tuple[np.ndarray, int]: A tuple containing:
            - log_mel_spectrogram (np.ndarray): Log-Mel spectrogram
            - num_frames (int): The number of frames in the spectrogram.
    """
    logger.info(
        f"Validating audio file: {file.filename}, "
        f"content type: {file.content_type}"
    )

    is_wav = (file.content_type == "audio/wav" or
              file.filename.lower().endswith(".wav"))
    if not is_wav:
        logger.warning(
            f"Invalid file type for {file.filename}. "
            f"Expected WAV, got {file.content_type}."
        )
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
            logger.error(
                f"Pydub could not process file {file.filename}: {e_pydub}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Could not process audio file. "
                    f"Ensure it is a valid audio format. Error: {e_pydub}"
                )
            )

        duration_s = len(audio_segment) / 1000.0
        logger.info(
            f"Audio file: {file.filename}, Duration: {duration_s:.2f}s, "
            f"Channels: {audio_segment.channels}"
        )

        duration_ok = MIN_DURATION_S <= duration_s <= MAX_DURATION_S
        if not duration_ok:
            logger.warning(
                f"Audio duration for {file.filename} ({duration_s:.2f}s) "
                f"out of range [{MIN_DURATION_S}-{MAX_DURATION_S}s]."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Audio duration must be between {MIN_DURATION_S} and "
                    f"{MAX_DURATION_S} seconds. Got {duration_s:.2f}s."
                )
            )

        if audio_segment.channels != 1:
            logger.info(f"Converting {file.filename} to mono.")
            audio_segment = audio_segment.set_channels(1)

        if audio_segment.frame_rate != TARGET_SAMPLE_RATE:
            logger.info(
                f"Resampling {file.filename} to {TARGET_SAMPLE_RATE}Hz "
                f"from {audio_segment.frame_rate}Hz."
            )
            audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE)

        processed_audio_stream = io.BytesIO()
        audio_segment.export(processed_audio_stream, format="wav")
        processed_audio_stream.seek(0)

        with sf.SoundFile(processed_audio_stream, 'r') as audio_sf_file:
            sample_rate = audio_sf_file.samplerate
            if sample_rate != TARGET_SAMPLE_RATE:
                logger.error(
                    f"Soundfile check failed: {file.filename} sample rate "
                    f"is {sample_rate}, expected {TARGET_SAMPLE_RATE}."
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal error: Audio sample rate mismatch."
                )
            if audio_sf_file.channels != 1:
                logger.error(
                    f"Soundfile check failed: {file.filename} channels is "
                    f"{audio_sf_file.channels}, expected 1."
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal error: Audio channel mismatch."
                )

            audio_data = audio_sf_file.read(dtype='float32')
            logger.info(
                f"Audio data stats: min={np.min(audio_data):.4f}, "
                f"max={np.max(audio_data):.4f}"
            )

            # --- Dithering Step ---
            rand_shape = audio_data.shape
            noise = DITHER_COEFF * np.random.randn(*rand_shape)
            audio_data_dithered = audio_data + noise
            # Clip after dither
            audio_data_dithered = np.clip(audio_data_dithered, -1.0, 1.0)
            logger.info(
                f"After dithering: min={np.min(audio_data_dithered):.4f}, "
                f"max={np.max(audio_data_dithered):.4f}"
            )

            # --- Pre-emphasis Filter ---
            first_sample = audio_data_dithered[0]
            rest = audio_data_dithered[1:]
            prev = audio_data_dithered[:-1]
            pre_emph_rest = rest - PREEMPHASIS_COEFF * prev
            audio_data_preemphasized = np.append(first_sample, pre_emph_rest)

            min_val = np.min(audio_data_preemphasized)
            max_val = np.max(audio_data_preemphasized)
            logger.info(
                f"After pre-emphasis: min={min_val:.4f}, max={max_val:.4f}"
            )

            # --- Mel Spectrogram Calculation ---
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data_preemphasized,
                sr=TARGET_SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                window='hann'  # Common window function
            )

            # --- Convert to Log-Mel Spectrogram (dB scale) ---
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # --- Per-feature Mean/Variance Normalization ---
            if log_mel.size > 0:
                # Calculate mean and std per feature
                mean = np.mean(log_mel, axis=1, keepdims=True)
                std_dev = np.std(log_mel, axis=1, keepdims=True)
                # Normalize
                epsilon = 1e-5  # To avoid division by zero
                log_mel = (log_mel - mean) / (std_dev + epsilon)
                min_spec = np.min(log_mel)
                max_spec = np.max(log_mel)
                logger.info(
                    f"Log-Mel stats: min={min_spec:.4f}, max={max_spec:.4f}"
                )
            else:
                logger.warning("Log-Mel spectrogram is empty.")
            num_frames = log_mel.shape[1]

        shape_str = str(log_mel.shape)
        logger.info(f"Generated log-Mel spectrogram. Shape: {shape_str}")
        return log_mel, num_frames

    except HTTPException:
        raise
    except sf.SoundFileError as e_sf:
        logger.error(
            f"Soundfile error for {file.filename}: {e_sf}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Could not read audio file. Ensure it is a valid WAV format. "
                f"Error: {str(e_sf)[:50]}"
            )
        )
    except Exception as e:
        logger.error(
            f"Error processing file {file.filename}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
