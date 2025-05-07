import cv2  # Or your preferred image loading library
import numpy as np

# --- Assume these predictor functions are imported correctly ---
from efficientnetb0 import predictor as predictor_efficientnet
from resnet50_predictor import predictor as predictor_resnet50
from densenet121.densenet import predictor as predictor_densenet
from swin.swin import predictor as predictor_swin
from yolo11s.yolo11s import predictor as predictor_yolo
from inceptionv3.inceptionv3 import predictor as predictor_inception

# --- Placeholder Predictor Functions (Replace with your actual imports) ---
# def predictor_efficientnet(img): return {'Atelectasis': 0.7, 'Cardiomegaly': 0.1, 'Effusion': 0.6, 'Nodule': 0.3, 'Pneumothorax': 0.8}
# def predictor_resnet50(img): return {'Atelectasis': 0.6, 'Cardiomegaly': 0.8, 'Effusion': 0.5, 'Nodule': 0.7, 'Pneumothorax': 0.2}
# def predictor_densenet(img): return {'Atelectasis': 0.65, 'Cardiomegaly': 0.75, 'Effusion': 0.8, 'Nodule': 0.75, 'Pneumothorax': 0.78}
# def predictor_swin(img): return {'Atelectasis': 0.5, 'Cardiomegaly': 0.85, 'Effusion': 0.6, 'Nodule': 0.65, 'Pneumothorax': 0.6}
# def predictor_yolo(img): return {'Atelectasis': 0.55, 'Cardiomegaly': 0.7, 'Effusion': 0.7, 'Nodule': 0.6, 'Pneumothorax': 0.65}
# def predictor_inception(img): return {'Atelectasis': 0.75, 'Cardiomegaly': 0.72, 'Effusion': 0.65, 'Nodule': 0.5, 'Pneumothorax': 0.7}
# --- End Placeholder Predictors ---

# --- List of classes ---
classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']

# --- Map model names to their predictor functions ---
model_predictors = {
    'EfficientNet': predictor_efficientnet,
    'ResNet50': predictor_resnet50,
    'DenseNet': predictor_densenet,
    'Swin Transformer': predictor_swin,
    'YOLOv11s': predictor_yolo,
    'InceptionV3': predictor_inception
}

def average_voting_ensemble_predict(image_path, threshold=0.5):
    """
    Performs plain average voting ensemble prediction.

    Args:
        image_path (str): Path to the input image.
        threshold (float): Threshold to convert average scores to binary predictions.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary of final average scores for each class.
            - dict: Dictionary of binary predictions (1 or 0) for each class.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        # Add any necessary preprocessing for your models here if needed
        # image = preprocess_image(image)

    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return None, None

    # --- Get predictions from all models ---
    model_predictions = {}
    print("Running individual model predictions...")
    for model_name, predictor_func in model_predictors.items():
        try:
            print(f"  Predicting with {model_name}...")
            model_predictions[model_name] = predictor_func(image)
            # Basic validation of prediction format
            if not isinstance(model_predictions[model_name], dict) or \
               not all(cls in model_predictions[model_name] for cls in classes):
                print(f"Warning: Unexpected output format from {model_name}. Expecting dict with keys: {classes}")
        except Exception as e:
            print(f"Error during prediction with {model_name}: {e}")
            continue
    print("Finished individual predictions.")

    # --- Calculate plain average scores (no weights) ---
    final_average_scores = {cls: 0.0 for cls in classes}

    for cls in classes:
        model_count = 0
        for model_name, individual_prediction in model_predictions.items():
            if cls in individual_prediction:
                prob = individual_prediction[cls]
                final_average_scores[cls] += prob
                model_count += 1

        if model_count > 0:
            final_average_scores[cls] /= model_count
        else:
            print(f"Warning: No predictions available for class '{cls}' from any model.")

    # --- Apply threshold for final binary predictions ---
    final_binary_predictions = {
        cls: 1 if final_average_scores[cls] >= threshold else 0
        for cls in classes
    }

    return final_average_scores, final_binary_predictions

# --- Example Usage ---
if __name__ == "__main__":
    image_file = r"C:\Users\arron\OneDrive\Documents\UTS\Post Graduate\Semester 4\49275 Neural Networks and Fuzzy Logic\Group Assignment\Datasets\1500_Xray\DATA\1500_train\test\Atelectasis\00000459_031.png" # <--- CHANGE THIS to your actual image path

    avg_scores, binary_predictions = average_voting_ensemble_predict(image_file)

    if avg_scores and binary_predictions:
        print("\n--- Ensemble Results ---")
        print("Final Average Scores:")
        for cls, score in avg_scores.items():
            print(f"  {cls}: {score:.4f}")

        print("\nFinal Binary Predictions (Threshold={}):".format(0.5))
        for cls, prediction in binary_predictions.items():
            print(f"  {cls}: {prediction}")
