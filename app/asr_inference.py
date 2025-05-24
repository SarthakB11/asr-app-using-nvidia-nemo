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

async def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribes audio data using the loaded ONNX ASR model."""
    if ort_session is None or input_name_signal is None or output_name_log_probs is None or ctc_labels is None or blank_label_idx == -1:
        logger.error("ASR model is not loaded or not configured properly. Cannot transcribe.")
        raise RuntimeError("ASR model is not ready for transcription.")

    try:
        chunk_size = 80  # Based on model's expected input shape
        num_samples = audio_data.shape[0]
        all_log_probs = []

        logger.info(f"Starting transcription for audio with {num_samples} samples, chunk size {chunk_size}.")

        for i in range(0, num_samples, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            actual_chunk_length = chunk.shape[0]

            if actual_chunk_length < chunk_size:
                # Pad the last chunk with zeros if it's smaller than chunk_size
                padding = np.zeros(chunk_size - actual_chunk_length, dtype=np.float32)
                chunk = np.concatenate((chunk, padding))
                logger.debug(f"Padded last chunk from {actual_chunk_length} to {chunk_size} samples.")
            
            # Reshape chunk to [1, chunk_size, 1] for [batch, samples, channels]
            audio_signal_chunk = chunk.astype(np.float32).reshape(1, chunk_size, 1)
            
            input_feed = {input_name_signal: audio_signal_chunk}

            if input_name_length:
                # The 'length' input for Conformer models usually refers to the number of valid frames/chunks,
                # not individual samples within the chunk, especially if the model processes fixed-size chunks internally.
                # For a fixed chunk_size of 80, the length input might be related to how many such 80-sample chunks are valid,
                # or it might expect the length of the signal *before* chunking if it handles chunking internally.
                # Given the error, the primary issue is the audio_signal dimension.
                # Let's assume for now the 'length' input refers to the number of samples in the *current* chunk being processed.
                # This might need adjustment if the model's 'length' input has a different meaning for chunked input.
                # For a fixed chunk size of 80, this length would always be 80.
                # CORRECTED: The length should be the number of acoustic frames after subsampling.
                # With subsampling_factor = 4, for a chunk_size of 80, length = 80 / 4 = 20.
                subsampling_factor = 4
                processed_length = chunk_size // subsampling_factor
                chunk_length_batch = np.array([processed_length], dtype=np.int64) 
                input_feed[input_name_length] = chunk_length_batch
                logger.debug(f"Processing chunk {i//chunk_size + 1}: signal shape {audio_signal_chunk.shape}, length {chunk_length_batch[0]} (after subsampling)")
            else:
                logger.debug(f"Processing chunk {i//chunk_size + 1}: signal shape {audio_signal_chunk.shape} (no length input)")

            def _run_inference_chunk(): # Closure to capture current chunk's input_feed
                return ort_session.run([output_name_log_probs], input_feed)
            
            log_probs_chunk_batch = await run_in_threadpool(_run_inference_chunk)
            # Assuming output is [batch_size, time_steps_in_chunk, num_classes]
            # For a single chunk in the batch, this is [1, time_steps_for_80_samples, num_classes]
            current_chunk_log_probs = log_probs_chunk_batch[0][0] 
            all_log_probs.append(current_chunk_log_probs)
            logger.debug(f"Chunk {i//chunk_size + 1} processed. Log probs shape: {current_chunk_log_probs.shape}")

        # Concatenate log_probs from all chunks along the time_steps axis (axis 0)
        if not all_log_probs:
            logger.warning("No log_probs were generated, possibly empty audio input.")
            return ""
            
        combined_log_probs = np.concatenate(all_log_probs, axis=0)
        logger.info(f"Combined log_probs shape: {combined_log_probs.shape}")

        transcription = _ctc_greedy_decode(combined_log_probs, ctc_labels)
        logger.info(f"Decoded transcription: {transcription}")
        
        return transcription

    except Exception as e:
        logger.error(f"Error during ASR transcription: {e}", exc_info=True)
        raise RuntimeError(f"ASR transcription failed: {e}")
