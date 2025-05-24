import onnxruntime as ort
import numpy as np
import logging
import json
from typing import List
from fastapi.concurrency import run_in_threadpool

# Import configuration for model path and vocab path
from .config import ONNX_MODEL_PATH, VOCAB_FILE_PATH

# Configure logger for this module
logger = logging.getLogger(__name__)

# Set logging level to DEBUG for detailed logging
logger.setLevel(logging.DEBUG)

# Add a basic handler directly to this logger for diagnostics
# Avoid adding duplicate basic handlers
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Revert to standard logger propagation
    logger.propagate = True

# Global variables for the ONNX session and model details
ort_session: ort.InferenceSession = None
input_name_signal: str = None
input_name_length: str = None
output_name_log_probs: str = None
ctc_labels: List[str] = None
blank_label_idx: int = -1

# Define N_MELS based on model expectation (from ONNX inspection)
N_MELS = 80


def load_model(model_path: str = ONNX_MODEL_PATH,
               vocab_path: str = VOCAB_FILE_PATH):
    """Loads the ONNX ASR model and its vocabulary."""
    global ort_session, input_name_signal, input_name_length
    global output_name_log_probs, ctc_labels, blank_label_idx

    if ort_session is not None:
        logger.info("ONNX model already loaded.")
        return

    try:
        logger.info(f"Loading ONNX ASR model from: {model_path}")
        options = ort.SessionOptions()
        # Set optimization level to maximum
        opt_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.graph_optimization_level = opt_level
        ort_session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=ort.get_available_providers()
        )

        inputs_meta = ort_session.get_inputs()
        outputs_meta = ort_session.get_outputs()

        if len(inputs_meta) < 1:
            raise RuntimeError(
                "ONNX model does not have the expected audio signal input."
            )

        input_name_signal = inputs_meta[0].name
        logger.info(
            f"Model Input (Signal): {input_name_signal}, "
            f"Shape: {inputs_meta[0].shape}, Type: {inputs_meta[0].type}"
        )

        if len(inputs_meta) > 1:
            input_name_length = inputs_meta[1].name
            logger.info(
                f"Model Input (Length): {input_name_length}, "
                f"Shape: {inputs_meta[1].shape}, Type: {inputs_meta[1].type}"
            )
        else:
            logger.warning(
                f"Model has only one input ({input_name_signal}). "
                "Assuming it does not require a separate length input. "
                "This might be an issue for some models."
            )
            input_name_length = None

        if not outputs_meta:
            raise RuntimeError("ONNX model does not have any outputs.")

        output_name_log_probs = outputs_meta[0].name
        logger.info(
            f"Model Output (Log Probs): {output_name_log_probs}, "
            f"Shape: {outputs_meta[0].shape}, Type: {outputs_meta[0].type}"
        )

        # Load vocabulary from JSON file
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                ctc_labels = json.load(f)

            # Validate vocabulary format
            if not isinstance(ctc_labels, list):
                raise ValueError("Vocab file did not contain a valid list.")

            vocab_size = len(ctc_labels)
            logger.info(
                f"Loaded vocabulary from {vocab_path} "
                f"({vocab_size} labels)"
            )

            # Determine blank_label_idx - assuming it's the last token
            # The vocabulary file contains only the actual characters.
            # Blank token is implicitly the class with index = len(ctc_labels)
            if ctc_labels:
                # Blank index is after all actual labels
                blank_label_idx = len(ctc_labels)
                # The character for blank is not in ctc_labels
                logger.info(f"Setting blank token index to {blank_label_idx}.")
            else:
                raise ValueError("Vocabulary is empty.")

        except FileNotFoundError:
            logger.error(f"Vocabulary file not found at: {vocab_path}")
            raise RuntimeError(f"Vocabulary file not found: {vocab_path}")
        except json.JSONDecodeError as e_json:
            logger.error(f"Error parsing vocabulary file: {e_json}")
            ort_session = None
            raise RuntimeError(
                f"Failed to parse vocabulary file {vocab_path}: {e_json}"
            )
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load vocabulary: {e}")

        logger.info("ONNX ASR model and vocabulary loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading model or vocabulary: {e}")
        ort_session = None
        raise RuntimeError(
            f"Failed to load ONNX ASR model: {e}"
        )


def _ctc_greedy_decode(log_probs: np.ndarray, labels: List[str]) -> str:
    """Performs CTC greedy decoding on the log probabilities.
    Args:
        log_probs: Numpy array of shape (time_steps, num_classes)
        labels: List of characters corresponding to class indices.
    Returns:
        A string of the decoded text.
    """
    logger.info(
        f"Effective log level for '{logger.name}': "
        f"{logging.getLevelName(logger.getEffectiveLevel())}"
    )

    if (log_probs is None or not isinstance(log_probs, np.ndarray) or
            log_probs.ndim != 2):
        logger.error(f"Invalid log_probs for CTC decoding: {type(log_probs)}")
        return "<error_decoding_invalid_input>"

    if not labels:
        logger.error("CTC labels not provided for decoding.")
        return "<error_decoding_no_labels>"

    # Use the global blank_label_idx, which should be set by load_model
    if blank_label_idx == -1:
        logger.error("Blank label index not set. Model might not be loaded.")
        return "<error_decoding_blank_idx_not_set>"

    logger.debug(f"CTC Decoding - Input log_probs shape: {log_probs.shape}")
    # Log sample of log_probs including the blank token's probability
    # First 10 or fewer
    sample_indices = list(range(min(10, log_probs.shape[1])))
    # Ensure blank_label_idx is a valid column index
    if log_probs.shape[1] > blank_label_idx:
        sample_indices.append(blank_label_idx)
    # Remove duplicates if blank_label_idx was already in the first 10
    sample_indices = sorted(list(set(sample_indices)))
    # Log a sample of the log probabilities for debugging
    if len(log_probs) > 0 and len(sample_indices) > 0:
        sample_row = log_probs[0]
        sample_probs = {i: sample_row[i] for i in sample_indices}
        logger.debug(
            f"CTC Decoding - Sample log probs (first row): {sample_probs}"
        )
    # Get the most likely class at each time step
    token_indices = np.argmax(log_probs, axis=1)
    logger.debug(f"CTC Decoding - Token Indices: {token_indices[:50]}")
    logger.debug(f"CTC Decoding - Blank Label Index: {blank_label_idx}")
    logger.debug(f"CTC Decoding - Vocabulary size: {len(labels)}")
    # For debugging, show the raw token sequence before collapsing
    raw_decoded_sequence = []
    for token_idx in token_indices:
        if token_idx == blank_label_idx:
            # Represent blank for clarity
            raw_decoded_sequence.append("<b>")
        elif 0 <= token_idx < len(labels):
            raw_decoded_sequence.append(labels[token_idx])
        else:
            raw_decoded_sequence.append("<unk>")
    logger.debug(
        f"CTC Decoding - Raw token sequence: "
        f"{''.join(raw_decoded_sequence[:100])}"
    )
    # Original CTC greedy decoding logic
    # List to store decoded tokens
    decoded_tokens = []
    # To track the last token for collapsing repeated tokens
    last_token = -1
    for token_idx in token_indices:
        # Skip blanks and repeats
        if token_idx != blank_label_idx and token_idx != last_token:
            # Ensure token index is valid for the labels list
            if token_idx < len(labels):
                decoded_tokens.append(labels[token_idx])
            else:
                # This path should ideally not be hit
                logger.warning(
                    f"Decoded token index {token_idx} out of bounds. "
                    f"Labels list size: {len(labels)}. Assigning <UNK>."
                )
                # Or perhaps skip this token
                decoded_tokens.append("<UNK>")
        last_token = token_idx
    final_transcription = "".join(decoded_tokens)
    logger.debug(
        f"CTC Decoding - Final transcription: '{final_transcription}'"
    )
    return final_transcription


async def transcribe_audio(log_mel_spectrogram: np.ndarray,
                           num_frames: int) -> str:
    """Transcribes audio data using the loaded ONNX ASR model."""
    logger.info(
        f"ASR_INFERENCE: Entered transcribe_audio. "
        f"Spectrogram shape: {log_mel_spectrogram.shape}, frames: {num_frames}"
    )

    # We're only reading these global variables, not modifying them
    # Using them directly without global declaration to avoid F824 warnings
    if ort_session is None:
        logger.error("ONNX model is not loaded. Call load_model() first.")
        raise RuntimeError("Model not loaded.")

    try:
        logger.info(
            f"Starting transcription for log-Mel spectrogram. "
            f"Shape: {log_mel_spectrogram.shape}, frames: {num_frames}"
        )

        # Ensure spectrogram has the expected number of Mel bins
        if log_mel_spectrogram.shape[0] != N_MELS:
            logger.error(
                f"Spectrogram has {log_mel_spectrogram.shape[0]} Mel bins, "
                f"but model expects {N_MELS}."
            )
            raise ValueError(
                f"Incorrect number of Mel bins. Expected {N_MELS}."
            )

        # Reshape spectrogram to [1, N_MELS, num_frames]
        # This matches the ONNX model's 'audio_signal' input
        audio_signal_input = log_mel_spectrogram.astype(np.float32).reshape(
            1, N_MELS, num_frames
        )

        input_feed = {input_name_signal: audio_signal_input}
        length_input_val_for_log = 'N/A'

        if input_name_length:
            # The 'length' input for the ONNX model is the number of frames
            length_input_arr = np.array([num_frames], dtype=np.int64)
            input_feed[input_name_length] = length_input_arr
            length_input_val_for_log = length_input_arr[0]

        shape_str = str(audio_signal_input.shape)
        logger.debug(
            f"ASR_INFERENCE: Preparing input feed. "
            f"Shape: {shape_str}, length: {length_input_val_for_log}"
        )

        # Define a synchronous wrapper for the inference call
        def _run_inference():
            logger.info("ASR_INFERENCE: Attempting ort_session.run()")
            result = ort_session.run([output_name_log_probs], input_feed)
            logger.info("ASR_INFERENCE: ort_session.run() completed.")
            return result

        # Run inference in a threadpool to avoid blocking the event loop
        log_probs_batch = await run_in_threadpool(_run_inference)

        # Output is [batch_size, time_steps_in_output, num_classes]
        # For a single item in the batch, this is:
        #   [1, time_steps_model_output, num_classes]
        # log_probs_batch[0] is tensor for first (and only) output name
        processed_log_probs = log_probs_batch[0][0]

        logger.info(
            f"ASR_INFERENCE: Inference complete. Shape: "
            f"{processed_log_probs.shape}"
        )

        logger.info("ASR_INFERENCE: Attempting CTC greedy decode.")
        transcription = _ctc_greedy_decode(processed_log_probs, ctc_labels)

        logger.info(
            f"ASR_INFERENCE: CTC greedy decode complete. "
            f"Transcription: {transcription}"
        )

        return transcription

    except Exception as e:
        logger.error(
            f"ASR_INFERENCE: Error during ASR transcription: {e}",
            exc_info=True
        )
        raise RuntimeError(f"ASR transcription failed: {e}")
