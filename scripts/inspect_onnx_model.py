# scripts/inspect_onnx_model.py
import onnxruntime as ort
import os

# Assuming the script is run from the project root
ONNX_MODEL_FILENAME = "stt_hi_conformer_ctc_medium.onnx"
MODELS_DIR = "models"
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, ONNX_MODEL_FILENAME)

def inspect_model(model_path):
    print(f"Attempting to load ONNX model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        inputs_meta = session.get_inputs()
        outputs_meta = session.get_outputs()

        print("\nModel Inputs:")
        for i, input_meta in enumerate(inputs_meta):
            print(f"  Input {i}:")
            print(f"    Name: {input_meta.name}")
            print(f"    Shape: {input_meta.shape} (Note: None or 'batch_size' often means dynamic batch size)")
            print(f"    Type: {input_meta.type}")

        print("\nModel Outputs:")
        for i, output_meta in enumerate(outputs_meta):
            print(f"  Output {i}:")
            print(f"    Name: {output_meta.name}")
            print(f"    Shape: {output_meta.shape}")
            print(f"    Type: {output_meta.type}")

    except Exception as e:
        print(f"An error occurred while inspecting the model: {e}")

if __name__ == "__main__":
    inspect_model(ONNX_MODEL_PATH)
