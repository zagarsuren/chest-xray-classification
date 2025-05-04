# predictor_yolo.py
import os
# Set environment variable to potentially reduce Ultralytics logging noise
# os.environ['YOLO_VERBOSE'] = 'False' # Uncomment if needed

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import torch # PyTorch is a dependency of ultralytics

# -----------------------------
# Configuration
# -----------------------------
# *** IMPORTANT: Set the correct path to your trained YOLO model weights file (.pt) ***
# This should be the 'best.pt' file from your training run's 'weights' directory.
MODEL_PATH = './weights/best.pt' # <<<--- UPDATE IF NEEDED

# Define expected class names (Optional but good for verification)
# The script will primarily rely on model.names after loading.
EXPECTED_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']

# -----------------------------
# Global Model Variable & Loading Logic
# -----------------------------
_model = None # Global variable to hold the loaded model
_model_class_names = None # Global variable to hold the model's class names

def load_yolo_model_once():
    """
    Loads the YOLO classification model structure and weights using the global _model variable.
    Ensures the model is loaded only once (lazy loading).

    Returns:
        tuple: (loaded_model, model_class_names) or (None, None) if loading fails.
    """
    global _model, _model_class_names
    # Check if model is already loaded
    if _model is None:
        print("YOLO model not loaded yet. Initializing and loading weights...")

        # Check if the specified weights file exists
        if not os.path.exists(MODEL_PATH):
            print(f"FATAL ERROR: YOLO model weights file not found at: '{MODEL_PATH}'")
            print("Please ensure the MODEL_PATH variable is correct and the file exists.")
            sys.exit(1) # Exit if weights are essential and missing

        # Load the trained YOLO model
        try:
            print(f"Loading YOLO model from: {MODEL_PATH}")
            # Pass the path to the .pt file directly to the YOLO constructor
            loaded_model = YOLO(MODEL_PATH)
            print("YOLO model loaded successfully.")

            # --- Verification ---
            # Get the class names the loaded model internally knows about
            internal_names = loaded_model.names
            # Convert the internal dictionary {index: name} to a list [name, name, ...]
            # sorted by index to ensure correct order.
            model_names_list = [internal_names[i] for i in sorted(internal_names.keys())]

            print(f"Model's internal classes count: {len(model_names_list)}")
            print(f"Model's internal class names (ordered): {model_names_list}")

            # Optional but recommended: Verify against your expected classes
            if len(model_names_list) != len(EXPECTED_CLASSES):
                 print(f"\n*** WARNING ***")
                 print(f"Mismatch in number of classes!")
                 print(f"Model reports {len(model_names_list)} classes: {model_names_list}")
                 print(f"Expected {len(EXPECTED_CLASSES)} classes: {EXPECTED_CLASSES}")
                 print(f"Predictions might be misaligned if the loaded model is incorrect.")
                 # Decide if this should be a fatal error:
                 # sys.exit(1)
            elif sorted(model_names_list) != sorted(EXPECTED_CLASSES): # Check if names match, ignoring order
                 print(f"\n*** WARNING ***")
                 print(f"Mismatch in class names!")
                 print(f"Model names: {model_names_list}")
                 print(f"Expected names: {EXPECTED_CLASSES}")
                 print(f"Check if the correct model was loaded and if EXPECTED_CLASSES is defined correctly.")
                 # Decide if this should be a fatal error:
                 # sys.exit(1)


            # Assign to global variables
            _model = loaded_model
            _model_class_names = model_names_list # Store the ordered list

        except Exception as e:
            print(f"FATAL ERROR: Failed to load YOLO model from {MODEL_PATH}.")
            print(f"Error details: {e}")
            # Ensure ultralytics and its dependencies (like PyTorch) are installed correctly.
            sys.exit(1) # Exit on loading failure
    # else:
        # print("YOLO model already loaded.") # Optional message

    # Return the loaded model and its class names
    return _model, _model_class_names

# -----------------------------
# Predictor Function Definition
# -----------------------------
def predictor(image_cv2):
    """
    Performs classification prediction on a single input image using the loaded YOLO model.

    Args:
        image_cv2: NumPy array representing the image (loaded using cv2).
                   Expected format is Height x Width x Channels (BGR).

    Returns:
        dict: A dictionary mapping class names (str) to their predicted
              softmax probabilities (float). Returns None if the input is invalid
              or if any step (loading, prediction) fails.
    """
    print("\n--- Starting YOLO Prediction ---")
    # --- Input Validation ---
    if image_cv2 is None:
        print("Prediction Error: Input image is None.")
        return None
    if not isinstance(image_cv2, np.ndarray):
        print("Prediction Error: Input image is not a NumPy array.")
        return None
    print(f"Input image shape: {image_cv2.shape}")

    # --- Ensure Model is Loaded ---
    model_loaded, model_class_names_loaded = load_yolo_model_once()
    if model_loaded is None or model_class_names_loaded is None:
         print("Prediction Error: YOLO model could not be loaded. Cannot proceed.")
         return None

    # --- Preprocessing (YOLO handles resizing, normalization internally, but requires RGB) ---
    print("Converting image BGR -> RGB for YOLO model...")
    # YOLO expects images in RGB format, while cv2 loads them in BGR.
    try:
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Prediction Error: Failed to convert image to RGB using cv2.cvtColor: {e}")
        return None

    # --- Perform Prediction ---
    print("Performing prediction using the YOLO model...")
    try:
        # Use model.predict(). It handles batching, resizing, normalization.
        # Pass the single RGB image directly. `verbose=False` reduces console output.
        # The result is a list containing one 'Results' object for the single image.
        results = model_loaded.predict(source=image_rgb, verbose=False)

        # --- Check Prediction Output ---
        if not results or len(results) == 0:
            print("Prediction Error: YOLO model returned no results.")
            return None

        # Get the Results object for the first (and only) image
        result = results[0]

        # Check if probabilities are available (standard for classification tasks)
        if hasattr(result, 'probs') and result.probs is not None:
            # Extract the probabilities tensor
            # result.probs.data is a torch tensor
            probabilities_tensor = result.probs.data

            # Move tensor to CPU (if it was on GPU) and convert to NumPy array
            softmax_probabilities = probabilities_tensor.cpu().numpy()

            # --- Verification ---
            if softmax_probabilities.shape != (len(model_class_names_loaded),):
                 print(f"Prediction Error: Unexpected prediction output shape. Expected ({len(model_class_names_loaded)},), but got {softmax_probabilities.shape}")
                 return None
            if not np.isclose(np.sum(softmax_probabilities), 1.0, atol=1e-5): # Slightly looser tolerance for YOLO maybe
                 print(f"Warning: Softmax probabilities do not sum close to 1 (Sum: {np.sum(softmax_probabilities)}).")

            # --- Format Output ---
            # Create the result dictionary using the model's internal class names (ordered)
            result_dict = {class_name: float(prob) for class_name, prob in zip(model_class_names_loaded, softmax_probabilities)}

            print("Prediction successful.")
            return result_dict # Return the formatted dictionary

        else:
            print("Prediction Error: Results object does not contain 'probs'. Is this a classification model?")
            print(f"Available attributes in result: {dir(result)}")
            return None

    # --- Error Handling for Prediction Step ---
    except Exception as e:
        print(f"Prediction Error: An exception occurred during model.predict(): {e}")
        # Consider logging the full traceback for detailed debugging
        # import traceback
        # traceback.print_exc()
        return None # Return None indicating prediction failure

# -----------------------------
# Example Usage (when script is run directly)
# -----------------------------
if __name__ == "__main__":
    print("\n=======================================")
    print("=== YOLO Predictor Script Example ===")
    print("=======================================")

    # --- Configuration for Example ---
    # *** IMPORTANT: Update this path to a valid test image in your environment ***
    EXAMPLE_IMAGE_PATH = '/home/sagemaker-user/AUT2025/X_ray/DATA/1500_train/test/Cardiomegaly/00003610_012.png' # <<<--- VERIFY THIS PATH

    print(f"\n--- Script Configuration ---")
    print(f"Using YOLO Model: {MODEL_PATH}")
    print(f"Expected Classes: {EXPECTED_CLASSES}")
    print(f"Example Image Path: {EXAMPLE_IMAGE_PATH}")

    # --- Check if example image exists ---
    if not os.path.exists(EXAMPLE_IMAGE_PATH):
        print(f"\n--- ERROR ---")
        print(f"Example image file not found: '{EXAMPLE_IMAGE_PATH}'")
        print("Please update the EXAMPLE_IMAGE_PATH variable.")
        print("---------------")
    else:
        # --- Load the example image using OpenCV ---
        print(f"\n--- Loading Image ---")
        input_image = cv2.imread(EXAMPLE_IMAGE_PATH)

        if input_image is None:
            print(f"Error: Failed to load image '{EXAMPLE_IMAGE_PATH}' using cv2.")
        else:
            print(f"Image loaded successfully.")
            print(f"Original image shape: {input_image.shape} (H, W, C - BGR)")

            # --- Call the predictor function ---
            prediction_result = predictor(input_image)

            # --- Display the results ---
            print("\n--- Prediction Results ---")
            if prediction_result is not None:
                print("Prediction successful. Results (Class: Probability):")
                max_prob = 0
                predicted_class = "N/A"
                # Print formatted dictionary
                for class_name, probability in prediction_result.items():
                    print(f"  - {class_name:<15}: {probability:.6f}")
                    if probability > max_prob:
                        max_prob = probability
                        predicted_class = class_name

                print(f"\n---> Highest Probability Class: {predicted_class} (Confidence: {max_prob:.4f})")
            else:
                print("Prediction failed. Check logs above for errors.")

    print("\n--- End of Example ---")