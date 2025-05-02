import cv2 # Or your preferred image loading library
import numpy as np # Useful for calculations
 
# --- Assume these predictor functions are imported correctly ---
# from efficientnet import predictor as predictor_efficientnet
# from resnet50 import predictor as predictor_resnet50 # Assuming corrected import
from densenet121.densenet import predictor as predictor_densenet
# from swin import predictor as predictor_swin
# from yolo import predictor as predictor_yolo
# from inception import predictor as predictor_inception
 

# --- End Placeholder Predictors ---
 
 
# --- Class-Specific Normalized Weights (from your table) ---
# Structure: weights[ClassName][ModelName] = weight
weights = {
    'Atelectasis': {
        'EfficientNet': 0.172, 'ResNet50': 0.160, 'DenseNet': 0.174,
        'Swin Transformer': 0.155, 'YOLOv11s': 0.157, 'InceptionV3': 0.182
    },
    'Cardiomegaly': {
        'EfficientNet': 0.166, 'ResNet50': 0.153, 'DenseNet': 0.166,
        'Swin Transformer': 0.175, 'YOLOv11s': 0.171, 'InceptionV3': 0.168
    },
    'Effusion': {
        'EfficientNet': 0.161, 'ResNet50': 0.163, 'DenseNet': 0.175,
        'Swin Transformer': 0.163, 'YOLOv11s': 0.171, 'InceptionV3': 0.166
    },
    'Nodule': {
        'EfficientNet': 0.159, 'ResNet50': 0.169, 'DenseNet': 0.185,
        'Swin Transformer': 0.164, 'YOLOv11s': 0.167, 'InceptionV3': 0.156
    },
    'Pneumothorax': {
        'EfficientNet': 0.174, 'ResNet50': 0.164, 'DenseNet': 0.176,
        'Swin Transformer': 0.151, 'YOLOv11s': 0.162, 'InceptionV3': 0.173
    }
}
 
# --- List of classes (ensure order matches predictor output if it's not a dict) ---
classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']
 
# --- Map model names to their predictor functions ---
# Make sure the keys here match the keys in the weights dictionary
model_predictors = {
    # 'EfficientNet': predictor_efficientnet,
    # 'ResNet50': predictor_resnet50, # Corrected name based on table
    'DenseNet': predictor_densenet,
    # 'Swin Transformer': predictor_swin,
    # 'YOLOv11s': predictor_yolo,
    # 'InceptionV3': predictor_inception
}
 
def weighted_ensemble_predict(image_path, threshold=0.5):
    """
    Performs weighted voting ensemble prediction based on class-specific weights.
 
    Args:
        image_path (str): Path to the input image.
        threshold (float): Threshold to convert weighted scores to binary predictions.
 
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary of final weighted scores for each class.
            - dict: Dictionary of binary predictions (1 or 0) for each class.
    """
    try:
        # Load the image (adjust loading method if necessary)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        # Add any necessary preprocessing for your models here
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
                 # Handle error appropriately, e.g., skip model or return error
        except Exception as e:
            print(f"Error during prediction with {model_name}: {e}")
            # Decide how to handle: skip model, return error, etc.
            # For now, we'll skip this model's contribution if it errors
            continue
    print("Finished individual predictions.")
 
    # --- Calculate weighted scores ---
    final_weighted_scores = {cls: 0.0 for cls in classes}
 
    for cls in classes:
        total_weight_used_for_class = 0.0 # Keep track of weights used in case a model failed
        for model_name, individual_prediction in model_predictions.items():
             # Check if this model successfully predicted and returned the expected class
            if model_name in weights[cls] and cls in individual_prediction:
                prob = individual_prediction[cls] # Probability from this model for this class
                weight = weights[cls][model_name] # Specific weight for this model and class
 
                final_weighted_scores[cls] += prob * weight
                total_weight_used_for_class += weight
            # else: Optional: print warning if model prediction is missing
 
        # Optional: Renormalize if some models failed and weights don't sum to ~1
        if total_weight_used_for_class > 0 and not np.isclose(total_weight_used_for_class, 1.0):
             print(f"Warning: Total weight used for class '{cls}' is {total_weight_used_for_class:.3f}. Renormalizing score.")
             final_weighted_scores[cls] /= total_weight_used_for_class
 
 
    # --- Apply threshold for final binary predictions ---
    final_binary_predictions = {
        cls: 1 if final_weighted_scores[cls] >= threshold else 0
        for cls in classes
    }
 
    return final_weighted_scores, final_binary_predictions
 
# --- Example Usage ---
if __name__ == "__main__":
    image_file = '/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/balanced_dataset_4/test/Cardiomegaly/00000032_053.png' 
 
    weighted_scores, binary_predictions = weighted_ensemble_predict(image_file)
 
    if weighted_scores and binary_predictions:
        print("\n--- Ensemble Results ---")
        print("Final Weighted Scores:")
        for cls, score in weighted_scores.items():
            print(f"  {cls}: {score:.4f}")
 
        print("\nFinal Binary Predictions (Threshold={}):".format(0.5)) # Assuming default threshold
        for cls, prediction in binary_predictions.items():
            print(f"  {cls}: {prediction}")