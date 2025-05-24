import nemo.collections.asr as nemo_asr
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
NEMO_MODEL_PATH = "/home/sarthak-bhardwaj/Documents/asr-app-using-nvidia-nemo/downloads/stt_hi_conformer_ctc_medium.nemo"
OUTPUT_ONNX_DIR = "/home/sarthak-bhardwaj/Documents/asr-app-using-nvidia-nemo/models"
ONNX_MODEL_FILENAME = "stt_hi_conformer_ctc_medium.onnx"
VOCAB_FILENAME = "vocabulary.json"
# --- End Configuration ---

def convert_nemo_to_onnx(nemo_path, output_dir, onnx_filename, vocab_filename):
    """
    Loads a .nemo ASR model, exports it to ONNX format, and saves its vocabulary.
    """
    logging.info(f"Attempting to load NeMo model from: {nemo_path}")
    if not os.path.exists(nemo_path):
        logging.error(f"NeMo model file not found at {nemo_path}. Please download it first.")
        logging.error(f"Download from: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium/files")
        return

    try:
        acoustic_model = nemo_asr.models.EncDecCTCModel.restore_from(nemo_path)
        logging.info("NeMo model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading NeMo model: {e}")
        logging.error("Please ensure nemo_toolkit[asr] is installed (pip install nemo_toolkit[asr]) and the .nemo file path is correct.")
        return

    os.makedirs(output_dir, exist_ok=True)
    onnx_export_path = os.path.join(output_dir, onnx_filename)

    logging.info(f"Exporting model to ONNX: {onnx_export_path}")
    try:
        # For some models, you might need to specify input_example or other args for export.
        # If export fails, consult NeMo documentation for the specific model.
        acoustic_model.export(onnx_export_path) 
        logging.info(f"Model successfully exported to {onnx_export_path}")
    except Exception as e:
        logging.error(f"Error exporting model to ONNX: {e}")
        logging.error("Ensure you have the necessary export dependencies (e.g., onnx, onnxruntime). You might need to check NeMo documentation for specific export parameters for this model type.")
        return

    # Extract and save vocabulary
    vocabulary = acoustic_model.decoder.vocabulary
    vocab_path = os.path.join(output_dir, vocab_filename)
    try:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(list(vocabulary), f, ensure_ascii=False, indent=4) # Convert to list
        logging.info(f"Vocabulary saved to: {vocab_path}")
        # Ensure vocabulary is treated as a list for subsequent operations too
        py_vocabulary = list(vocabulary) 
        logging.info(f"Vocabulary (first 10 items): {py_vocabulary[:10]}")
        logging.info(f"Total vocabulary size: {len(py_vocabulary)}")
        
        blank_token = "<blank>"
        blank_idx_found = -1
        try:
            # Standard way to get blank_id in newer NeMo versions
            blank_idx_found = acoustic_model.tokenizer.blank
        except AttributeError:
            # Fallback for older versions or different tokenizer structures
            if hasattr(acoustic_model.decoder, 'blank_idx'):
                 blank_idx_found = acoustic_model.decoder.blank_idx
            elif len(py_vocabulary) > 0 and py_vocabulary[-1] == blank_token: # Common convention
                blank_idx_found = len(py_vocabulary) - 1
            elif blank_token in py_vocabulary:
                blank_idx_found = py_vocabulary.index(blank_token)

        if blank_idx_found != -1 and blank_idx_found < len(py_vocabulary):
            logging.info(f"Blank token: '{py_vocabulary[blank_idx_found]}' found at index: {blank_idx_found}")
        else:
            # If blank_idx is not found, it might be implicitly handled or needs specific model knowledge
            # For many CTC models, the blank is often the last index if not explicitly defined.
            assumed_blank_idx = len(py_vocabulary) -1 
            logging.warning(f"Could not definitively determine the blank token or its index. Assuming it might be the last token: '{py_vocabulary[assumed_blank_idx if assumed_blank_idx < len(py_vocabulary) else 'N/A']}' at index {assumed_blank_idx}. Manual verification might be needed for CTC decoding.")

    except Exception as e:
        logging.error(f"Error saving vocabulary: {e}")

if __name__ == "__main__":
    logging.info("Starting NeMo to ONNX conversion script.")
    # --- Create directories if they don't exist ---
    # Ensure download directory exists (though user needs to place file there)
    nemo_model_dir = os.path.dirname(NEMO_MODEL_PATH)
    if nemo_model_dir and not os.path.exists(nemo_model_dir):
        os.makedirs(nemo_model_dir, exist_ok=True)
        logging.info(f"Created directory for NeMo model: {nemo_model_dir}")
    
    # Ensure output ONNX directory exists
    if not os.path.exists(OUTPUT_ONNX_DIR):
        os.makedirs(OUTPUT_ONNX_DIR, exist_ok=True)
        logging.info(f"Created directory for ONNX output: {OUTPUT_ONNX_DIR}")

    convert_nemo_to_onnx(NEMO_MODEL_PATH, OUTPUT_ONNX_DIR, ONNX_MODEL_FILENAME, VOCAB_FILENAME)
    logging.info("Conversion script finished.")
