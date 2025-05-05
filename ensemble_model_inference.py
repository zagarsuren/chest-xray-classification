import cv2 # Or your preferred image loading library
import numpy as np # Useful for calculations
import importlib.util
import sys
import types
 
# --- Import the original modules ---
from efficientnetb0.efficientNet import load_bestmodel as load_efficientnet, predictor as predictor_efficientnet
from densenet121.densenet import load_model as load_densenet, predictor as predictor_densenet
from swin.swin import predictor as predictor_swin
from yolo11s.YOLO11s import load_yolo_model_once, predictor as predictor_yolo
from inceptionv3.inceptionNet import load_bestmodel as load_inception, predictor as predictor_inception
 

class EnsembleModelClassifier:
    """
    A classifier that ensembles multiple chest X-ray classification models.
    Models are loaded once at initialization and reused for all predictions.
    """
    
    def __init__(self):
        """Initialize the ensemble classifier."""
        # Class-Specific Normalized Weights (from your table)
        # Structure: weights[ClassName][ModelName] = weight
        self.weights = {
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
        
        # List of classes (ensure order matches predictor output if it's not a dict)
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']
        
        # Models will be stored here
        self.initialized_models = {}
        
        # Load all models at initialization
        self.initialize_models()
    
    def initialize_models(self):
        """Load all model weights at the beginning"""
        print("Loading model weights...")
        
        try:
            # Load EfficientNet model
            print("  Loading EfficientNet model...")
            efficientnet_model = load_efficientnet()
            self.initialized_models['EfficientNet'] = efficientnet_model
            
            # Load DenseNet model
            print("  Loading DenseNet model...")
            densenet_model = load_densenet()
            self.initialized_models['DenseNet'] = densenet_model
            
            # Load InceptionV3 model
            print("  Loading InceptionV3 model...")
            inception_model = load_inception()
            self.initialized_models['InceptionV3'] = inception_model
            
            # Load YOLO model
            print("  Loading YOLO model...")
            yolo_model, yolo_class_names = load_yolo_model_once()
            self.initialized_models['YOLOv11s'] = (yolo_model, yolo_class_names)
            
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error during model initialization: {e}")
    
    def predict(self, image_path, method="weighted_voting"):
        """
        Performs prediction based on the specified method.
    
        Args:
            image_path (str): Path to the input image.
            method (str): The voting method to use - either "weighted_voting" or "voting".
    
        Returns:
            tuple: A tuple containing:
                - dict: Dictionary of final scores for each class (weighted or vote counts).
                - str: Predicted class with the highest probability or vote count.
        """
        try:
            # Load the image (adjust loading method if necessary)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
    
        except Exception as e:
            print(f"Error loading or preprocessing image: {e}")
            return None, None
    
        # --- Get predictions from all models ---
        model_predictions = {}
        print("Running individual model predictions...")
        
        # EfficientNet prediction
        try:
            print("  Predicting with EfficientNet...")
            model_predictions['EfficientNet'] = predictor_efficientnet(image, self.initialized_models['EfficientNet'])
        except Exception as e:
            print(f"Error during prediction with EfficientNet: {e}")
        
        # DenseNet prediction
        try:
            print("  Predicting with DenseNet...")
            model_predictions['DenseNet'] = predictor_densenet(image, self.initialized_models['DenseNet'])
        except Exception as e:
            print(f"Error during prediction with DenseNet: {e}")
        
        # InceptionV3 prediction
        try:
            print("  Predicting with InceptionV3...")
            model_predictions['InceptionV3'] = predictor_inception(image, self.initialized_models['InceptionV3'])
        except Exception as e:
            print(f"Error during prediction with InceptionV3: {e}")
        
        # YOLO prediction
        try:
            print("  Predicting with YOLOv11s...")
            yolo_model, yolo_class_names = self.initialized_models['YOLOv11s']
            model_predictions['YOLOv11s'] = predictor_yolo(image, yolo_model, yolo_class_names)
        except Exception as e:
            print(f"Error during prediction with YOLOv11s: {e}")
        
        # Swin prediction (doesn't have a pre-loaded model yet)
        try:
            print("  Predicting with Swin Transformer...")
            model_predictions['Swin Transformer'] = predictor_swin(image)
        except Exception as e:
            print(f"Error during prediction with Swin Transformer: {e}")
        
        print("Finished individual predictions.")
    
        if method == "weighted_voting":
            # --- Calculate weighted scores ---
            final_scores = {cls: 0.0 for cls in self.classes}
        
            for cls in self.classes:
                total_weight_used_for_class = 0.0 # Keep track of weights used in case a model failed
                for model_name, individual_prediction in model_predictions.items():
                    # Check if this model successfully predicted and returned the expected class
                    if model_name in self.weights[cls] and cls in individual_prediction:
                        prob = individual_prediction[cls] # Probability from this model for this class
                        weight = self.weights[cls][model_name] # Specific weight for this model and class
            
                        final_scores[cls] += prob * weight
                        total_weight_used_for_class += weight
                    # else: Optional: print warning if model prediction is missing
            
                # Optional: Renormalize if some models failed and weights don't sum to ~1
                if total_weight_used_for_class > 0 and not np.isclose(total_weight_used_for_class, 1.0):
                    print(f"Warning: Total weight used for class '{cls}' is {total_weight_used_for_class:.3f}. Renormalizing score.")
                    final_scores[cls] /= total_weight_used_for_class
        
        elif method == "voting":
            # --- Calculate normal voting scores (majority voting) ---
            final_scores = {cls: 0 for cls in self.classes}
            
            # For each model, find the class with highest probability and count it as a vote
            for model_name, individual_prediction in model_predictions.items():
                if individual_prediction:  # Make sure the model prediction is available
                    # Find class with highest probability for this model
                    predicted_class = max(individual_prediction, key=individual_prediction.get)
                    # Add a vote for this class
                    final_scores[predicted_class] += 1
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'weighted_voting' or 'voting'.")
    
        # --- Use max to find the class with the highest score/vote ---
        predicted_class = max(final_scores, key=final_scores.get)
    
        return final_scores, predicted_class


# --- Example Usage ---
if __name__ == "__main__":
    # Create the ensemble classifier
    ensemble = EnsembleModelClassifier()
    
    # Test image path
    image_file = '/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/balanced_dataset_4/test/Cardiomegaly/00000032_053.png' 
    
    # Demonstrate weighted voting method
    print("\n=== Testing Weighted Voting Method ===")
    weighted_scores, weighted_prediction = ensemble.predict(image_file, method="weighted_voting")
 
    if weighted_scores and weighted_prediction:
        print("\n--- Weighted Voting Results ---")
        print("Final Weighted Scores:")
        for cls, score in weighted_scores.items():
            print(f"  {cls}: {score:.4f}")
 
        print("\nPredicted Class (weighted voting):")
        print(f"  {weighted_prediction} (score: {weighted_scores[weighted_prediction]:.4f})")
    
    # Demonstrate normal voting method
    print("\n=== Testing Normal Voting Method ===")
    vote_counts, voting_prediction = ensemble.predict(image_file, method="voting")
    
    if vote_counts and voting_prediction:
        print("\n--- Normal Voting Results ---")
        print("Vote Counts:")
        for cls, count in vote_counts.items():
            print(f"  {cls}: {count}")
        
        print("\nPredicted Class (majority voting):")
        print(f"  {voting_prediction} (votes: {vote_counts[voting_prediction]})")