import nemo.collections.asr as nemo_asr
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
BASE_DIR = "."
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
MODELS_DIR = os.path.join(BASE_DIR, "models")

NEMO_MODEL_FILENAME = "stt_hi_conformer_ctc_medium.nemo"
ONNX_MODEL_FILENAME = "stt_hi_conformer_ctc_medium.onnx"
VOCAB_FILENAME = "vocabulary.json"

NEMO_MODEL_PATH = os.path.join(DOWNLOADS_DIR, NEMO_MODEL_FILENAME)
# --- End Configuration ---


def convert_nemo_to_onnx(nemo_path, output_dir, onnx_filename, vocab_filename):
    """
    Loads a .nemo ASR model, exports it to ONNX format, and saves vocabulary.
    """
    model_name = os.path.basename(nemo_path)
    logging.info(f"Loading NeMo model: {model_name}")
    if not os.path.exists(nemo_path):
        logging.error(f"Model not found: {model_name}")
        logging.error(
            "Download from: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/"
            "nemo/models/stt_hi_conformer_ctc_medium/files"
        )
        return

    try:
        model = nemo_asr.models.EncDecCTCModel.restore_from(nemo_path)
        logging.info("NeMo model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading NeMo model: {e}")
        logging.error(
            "Please ensure nemo_toolkit[asr] is installed "
            "and the .nemo file path is correct."
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    onnx_export_path = os.path.join(output_dir, onnx_filename)

    logging.info("Exporting model to ONNX...")
    try:
        # For some models, you might need to specify input_example
        model.export(onnx_export_path)
        out_name = os.path.basename(onnx_export_path)
        logging.info(f"Model exported to: {out_name}")
    except Exception as e:
        logging.error(f"Error exporting model to ONNX: {e}")
        logging.error(
            "Ensure you have the necessary export dependencies. "
            "Check NeMo documentation for specific parameters."
        )
        return

    # Extract and save vocabulary
    vocabulary = model.decoder.vocabulary
    vocab_path = os.path.join(output_dir, vocab_filename)
    try:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            # Convert to list
            json.dump(list(vocabulary), f, ensure_ascii=False, indent=4)
        logging.info(f"Vocabulary saved to: {vocab_path}")

        # Ensure vocabulary is treated as a list for subsequent operations
        py_vocabulary = list(vocabulary)
        vocab_size = len(py_vocabulary)
        logging.info(f"First 10 vocab items: {py_vocabulary[:10]}")
        logging.info(f"Total vocabulary size: {vocab_size}")

        blank_token = "<blank>"
        blank_idx_found = -1
        try:
            # Standard way to get blank_id in newer NeMo versions
            blank_idx_found = model.tokenizer.blank
        except AttributeError:
            # Fallback for older versions or different tokenizer structures
            if hasattr(model.decoder, 'blank_idx'):
                blank_idx_found = model.decoder.blank_idx
            # Common convention
            elif vocab_size > 0 and py_vocabulary[-1] == blank_token:
                blank_idx_found = vocab_size - 1
            elif blank_token in py_vocabulary:
                blank_idx_found = py_vocabulary.index(blank_token)

        if blank_idx_found != -1 and blank_idx_found < vocab_size:
            logging.info(
                f"Blank token: '{py_vocabulary[blank_idx_found]}' "
                f"at index: {blank_idx_found}"
            )
        else:
            # If blank_idx is not found, it might be implicitly handled
            assumed_blank_idx = vocab_size - 1
            last_token = "N/A"
            if assumed_blank_idx < vocab_size:
                last_token = py_vocabulary[assumed_blank_idx]
            logging.warning(
                "Could not determine the blank token. "
                f"Assuming: '{last_token}' at index {assumed_blank_idx}."
            )

    except Exception as e:
        logging.error(f"Error saving vocabulary: {e}")


if __name__ == "__main__":
    logging.info("Starting NeMo to ONNX conversion script.")
    # Ensure download directory exists
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        logging.info(f"Created downloads directory: {DOWNLOADS_DIR}")

    # Ensure output ONNX directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        logging.info(f"Created models directory: {MODELS_DIR}")

    convert_nemo_to_onnx(
        NEMO_MODEL_PATH, MODELS_DIR, ONNX_MODEL_FILENAME, VOCAB_FILENAME
    )
    logging.info("Conversion script finished.")
