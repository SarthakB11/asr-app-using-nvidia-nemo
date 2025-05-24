import onnxruntime as ort
import numpy as np
import logging
import json
from typing import List
from fastapi.concurrency import run_in_threadpool

# Import configuration for model path and vocab path
from .config import ONNX_MODEL_PATH, VOCAB_FILE_PATH # Removed LOG_LEVEL import for now

# Configure logger for this module
logger = logging.getLogger(__name__) # This will be 'asr_app.asr_inference'
logger.setLevel(logging.DEBUG) # Set level to DEBUG; relies on parent handler (from main.py) and Uvicorn log level for output.

# Add a basic handler directly to this logger for diagnostics
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers): # Avoid adding duplicate basic handlers
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = True # Revert to standard logger propagation

# Global variables for the ONNX session and model details
ort_session: ort.InferenceSession = None
input_name_signal: str = None
input_name_length: str = None
output_name_log_probs: str = None
ctc_labels: List[str] = None
blank_label_idx: int = -1

# Define N_MELS based on model expectation (from ONNX inspection)
N_MELS = 80 

def load_model(model_path: str = ONNX_MODEL_PATH, vocab_path: str = VOCAB_FILE_PATH):
    """Loads the ONNX ASR model and its vocabulary."""
    global ort_session, input_name_signal, input_name_length, output_name_log_probs, ctc_labels, blank_label_idx

    if ort_session is not None:
        logger.info("ONNX model already loaded.")
        return

    try:
        logger.info(f"Loading ONNX ASR model from: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        ort_session = ort.InferenceSession(model_path, sess_options=options, providers=ort.get_available_providers())
        
        inputs_meta = ort_session.get_inputs()
        outputs_meta = ort_session.get_outputs()

        if len(inputs_meta) < 1:
            raise RuntimeError("ONNX model does not have the expected audio signal input.")
        
        input_name_signal = inputs_meta[0].name
        logger.info(f"Model Input (Signal): {input_name_signal}, Shape: {inputs_meta[0].shape}, Type: {inputs_meta[0].type}")
        
        if len(inputs_meta) > 1:
            input_name_length = inputs_meta[1].name
            logger.info(f"Model Input (Length): {input_name_length}, Shape: {inputs_meta[1].shape}, Type: {inputs_meta[1].type}")
        else:
            logger.warning(f"Model has only one input ({input_name_signal}). Assuming it does not require a separate length input. This might be an issue for some models.")
            input_name_length = None

        if not outputs_meta:
            raise RuntimeError("ONNX model does not have any outputs.")
        output_name_log_probs = outputs_meta[0].name
        logger.info(f"Model Output (Log Probs): {output_name_log_probs}, Shape: {outputs_meta[0].shape}, Type: {outputs_meta[0].type}")

        # Load vocabulary from JSON file
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                ctc_labels = json.load(f)
            if not isinstance(ctc_labels, list):
                raise ValueError("Vocabulary file did not contain a valid list.")
            logger.info(f"Successfully loaded vocabulary from {vocab_path} with {len(ctc_labels)} labels.")

            # Determine blank_label_idx - assuming it's the last token as per NeMo convention / conversion script warning
            # The vocabulary file (ctc_labels) contains only the actual characters.
            # The blank token is implicitly the class with index = len(ctc_labels).
            if ctc_labels:
                blank_label_idx = len(ctc_labels) # Corrected: blank index is after all actual labels
                # The character for blank is not in ctc_labels, so logging ctc_labels[blank_label_idx] would be an error.
                logger.info(f"Setting blank token index to {blank_label_idx}.") 
            else:
                raise ValueError("Vocabulary is empty.")

        except FileNotFoundError:
            logger.error(f"Vocabulary file not found at: {vocab_path}")
            raise RuntimeError(f"Vocabulary file not found: {vocab_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from vocabulary file: {vocab_path}")
            raise RuntimeError(f"Invalid JSON in vocabulary file: {vocab_path}")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load vocabulary: {e}")

        logger.info(f"ONNX ASR model and vocabulary loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading ONNX ASR model: {e}", exc_info=True)
        ort_session = None
        raise RuntimeError(f"Failed to load ONNX ASR model from {model_path}: {e}")

def _ctc_greedy_decode(log_probs: np.ndarray, labels: List[str]) -> str:
    """Performs CTC greedy decoding on the log probabilities.
    log_probs: Numpy array of shape (time_steps, num_classes)
    labels: List of characters corresponding to class indices.
    """
    logger.info(f"Effective log level for '{logger.name}': {logging.getLevelName(logger.getEffectiveLevel())}")

    if log_probs is None or not isinstance(log_probs, np.ndarray) or log_probs.ndim != 2:
        logger.error(f"Invalid log_probs for CTC decoding: {type(log_probs)}")
        return "<error_decoding_invalid_input>"
    if not labels:
        logger.error("CTC labels not provided for decoding.")
        return "<error_decoding_no_labels>"
    
    global blank_label_idx
    if blank_label_idx == -1:
        logger.error("Blank label index not set. Model might not be loaded correctly.")
        return "<error_decoding_blank_idx_not_set>"

    logger.debug(f"CTC Decoding - Input log_probs shape: {log_probs.shape}")
    # Log sample of log_probs including the blank token's probability
    sample_indices = list(range(min(10, log_probs.shape[1]))) # First 10 or fewer
    if log_probs.shape[1] > blank_label_idx: # Ensure blank_label_idx is a valid column index
        sample_indices.append(blank_label_idx)
    # Remove duplicates if blank_label_idx was already in the first 10 and sort for consistent view
    sample_indices = sorted(list(set(sample_indices)))

    logger.debug(f"CTC Decoding - Log Probs (sample, first 3 rows, cols {sample_indices} incl blank at {blank_label_idx}):\n{log_probs[:3, sample_indices]}")
    
    token_indices = np.argmax(log_probs, axis=1)
    logger.debug(f"CTC Decoding - Token Indices (first 50): {token_indices[:50]}")
    logger.debug(f"CTC Decoding - Blank Label Index used: {blank_label_idx}")
    logger.debug(f"CTC Decoding - Vocabulary size (len(labels)): {len(labels)}")

    # Log tokens before CTC collapse and blank removal for debugging
    raw_decoded_sequence = []
    for token_idx in token_indices:
        if token_idx == blank_label_idx:
            raw_decoded_sequence.append("<b>") # Represent blank for clarity
        elif 0 <= token_idx < len(labels):
            raw_decoded_sequence.append(labels[token_idx])
        else:
            raw_decoded_sequence.append(f"<INVALID_IDX_{token_idx}>")
    logger.debug(f"CTC Decoding - Raw token sequence (first 100 symbols): {''.join(raw_decoded_sequence[:100])}")

    # Original CTC greedy decoding logic
    decoded_tokens = []
    last_token = None

    for token_idx in token_indices:
        if token_idx == blank_label_idx:
            last_token = None
            continue
        if token_idx == last_token:
            continue
        
        if 0 <= token_idx < len(labels):
            decoded_tokens.append(labels[token_idx])
        else:
            # This path should ideally not be hit if blank_label_idx is len(labels)
            # and token_idx is not blank. It implies token_idx > len(labels).
            logger.warning(f"Decoded token index {token_idx} is out of bounds for labels list (size {len(labels)}). Assigning <UNK>.")
            decoded_tokens.append("<UNK>") # Or perhaps skip this token
        last_token = token_idx
        
    final_transcription = "".join(decoded_tokens)
    logger.debug(f"CTC Decoding - Final transcription after collapse: '{final_transcription}'")
    return final_transcription

async def transcribe_audio(log_mel_spectrogram: np.ndarray, num_frames: int) -> str:
    """Transcribes audio data (as log-Mel spectrogram) using the loaded ONNX ASR model."""
    logger.info(f"ASR_INFERENCE: Entered transcribe_audio. Spectrogram shape: {log_mel_spectrogram.shape}, num_frames: {num_frames}")
    global ort_session, input_name_signal, input_name_length, output_name_log_probs, ctc_labels

    if ort_session is None:
        logger.error("ONNX model is not loaded. Call load_model() first.")
        raise RuntimeError("Model not loaded.")

    try:
        logger.info(f"Starting transcription for log-Mel spectrogram with shape {log_mel_spectrogram.shape} ({num_frames} frames).")

        # Ensure spectrogram has the expected number of Mel bins
        if log_mel_spectrogram.shape[0] != N_MELS:
            logger.error(f"Spectrogram has {log_mel_spectrogram.shape[0]} Mel bins, but model expects {N_MELS}.")
            raise ValueError(f"Incorrect number of Mel bins. Expected {N_MELS}.")

        # Reshape spectrogram to [1, N_MELS, num_frames] for [batch, features, time_steps]
        # This matches the ONNX model's 'audio_signal' input: (batch_size, 80, num_frames)
        audio_signal_input = log_mel_spectrogram.astype(np.float32).reshape(1, N_MELS, num_frames)
        
        input_feed = {input_name_signal: audio_signal_input}
        length_input_val_for_log = 'N/A'

        if input_name_length:
            # The 'length' input for the ONNX model is the number of frames in the spectrogram
            length_input_arr = np.array([num_frames], dtype=np.int64)
            input_feed[input_name_length] = length_input_arr
            length_input_val_for_log = length_input_arr[0]
        
        logger.debug(f"ASR_INFERENCE: Preparing input feed. Signal shape {audio_signal_input.shape}, length {length_input_val_for_log}")

        # Define a synchronous wrapper for the inference call to run in threadpool
        def _run_inference(): 
            logger.info("ASR_INFERENCE: Attempting ort_session.run()")
            result = ort_session.run([output_name_log_probs], input_feed)
            logger.info("ASR_INFERENCE: ort_session.run() completed.")
            return result
        
        # Run inference in a threadpool to avoid blocking the event loop
        log_probs_batch = await run_in_threadpool(_run_inference)
        
        # Output is [batch_size, time_steps_in_output, num_classes]
        # For a single item in the batch, this is [1, time_steps_model_output, num_classes]
        # We need to pass [time_steps_model_output, num_classes] to _ctc_greedy_decode
        # log_probs_batch[0] is the tensor for the first (and only) output name
        # log_probs_batch[0][0] extracts the actual log_probs for the first batch item
        processed_log_probs = log_probs_batch[0][0]
        
        logger.info(f"ASR_INFERENCE: Inference complete. Log probs shape: {processed_log_probs.shape}")

        logger.info("ASR_INFERENCE: Attempting CTC greedy decode.")
        transcription = _ctc_greedy_decode(processed_log_probs, ctc_labels)
        logger.info(f"ASR_INFERENCE: CTC greedy decode complete. Transcription: {transcription}")
        
        return transcription

    except Exception as e:
        logger.error(f"ASR_INFERENCE: Error during ASR transcription: {e}", exc_info=True)
        raise RuntimeError(f"ASR transcription failed: {e}")
