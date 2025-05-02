import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
import json
import glob

# Constants
IMG_SIZE = (512, 512)
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Nodule', 'Pneumothorax', 'Effusion']
NUM_CLASSES = len(CLASSES)
MODEL_WEIGHTS_PATH = './densenet121/best_model'

# Function to create the model architecture for loading weights
def create_base_model(l2_reg=1e-4, dropout_rate=0.5, dense_units_1=1024, dense_units_2=512, dense_units_3=256, learning_rate=1e-4):
    """
    Create a DenseNet121 model with customizable hyperparameters.
    
    Args:
        l2_reg: L2 regularization strength
        dropout_rate: Dropout rate for regularization
        dense_units_1: Number of units in first dense layer
        dense_units_2: Number of units in second dense layer
        dense_units_3: Number of units in third dense layer
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units_3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load the model hyperparameters and weights
def load_model():
    try:
        # Try to load hyperparameters from trial.json
        if os.path.exists(os.path.join(MODEL_WEIGHTS_PATH, 'trial.json')):
            with open(os.path.join(MODEL_WEIGHTS_PATH, 'trial.json'), 'r') as f:
                trial_data = json.load(f)
            
            # Extract hyperparameters
            hyperparameters = trial_data.get('hyperparameters', {}).get('values', {})
            l2_reg = hyperparameters.get('l2_reg', 1e-4)
            dropout_rate = hyperparameters.get('dropout_rate', 0.5)
            dense_units_1 = hyperparameters.get('dense_units_1', 1024)
            dense_units_2 = hyperparameters.get('dense_units_2', 512)
            dense_units_3 = hyperparameters.get('dense_units_3', 256)
            learning_rate = hyperparameters.get('learning_rate', 1e-4)
        else:
            # Look for trial directories in the parent directory
            parent_dir = os.path.dirname(MODEL_WEIGHTS_PATH)
            trial_dirs = glob.glob(os.path.join(parent_dir, 'trial_*'))
            
            if not trial_dirs:
                # If no trial directories found, try to use the parent directory
                trial_dirs = glob.glob(os.path.join(os.path.dirname(parent_dir), 'trial_*'))
            
            if trial_dirs:
                # Find the best trial based on score
                best_score = -float('inf')
                best_hyperparams = None
                
                for trial_dir in trial_dirs:
                    trial_path = os.path.join(trial_dir, 'trial.json')
                    
                    if os.path.exists(trial_path):
                        with open(trial_path, 'r') as f:
                            trial_data = json.load(f)
                        
                        score = trial_data.get('score', 0)
                        
                        if score > best_score:
                            best_score = score
                            best_hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
                
                if best_hyperparams:
                    # Extract hyperparameters from best trial
                    l2_reg = best_hyperparams.get('l2_reg', 1e-4)
                    dropout_rate = best_hyperparams.get('dropout_rate', 0.5)
                    dense_units_1 = best_hyperparams.get('dense_units_1', 1024)
                    dense_units_2 = best_hyperparams.get('dense_units_2', 512)
                    dense_units_3 = best_hyperparams.get('dense_units_3', 256)
                    learning_rate = best_hyperparams.get('learning_rate', 1e-4)
                else:
                    # Default values
                    l2_reg, dropout_rate = 1e-4, 0.5
                    dense_units_1, dense_units_2, dense_units_3 = 1024, 512, 256
                    learning_rate = 1e-4
            else:
                # Default values
                l2_reg, dropout_rate = 1e-4, 0.5
                dense_units_1, dense_units_2, dense_units_3 = 1024, 512, 256
                learning_rate = 1e-4
    except Exception as e:
        # Default values
        l2_reg, dropout_rate = 1e-4, 0.5
        dense_units_1, dense_units_2, dense_units_3 = 1024, 512, 256
        learning_rate = 1e-4
    
    # Create model with hyperparameters
    model = create_base_model(
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2, 
        dense_units_3=dense_units_3,
        learning_rate=learning_rate
    )
    
    # Load weights
    weights_path = os.path.join(MODEL_WEIGHTS_PATH, 'checkpoint.weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")
    
    model.load_weights(weights_path)
    return model

# Global model variable to avoid reloading
_model = None

def predictor(img):
    """
    Predict chest X-ray conditions from an OpenCV image.
    
    Args:
        img: Numpy array containing the image (BGR format from OpenCV)
        
    Returns:
        Dictionary mapping class names to confidence scores
    """
    global _model
    
    # Load model if not already loaded
    if _model is None:
        _model = load_model()
    
    # Preprocess the image
    # Convert BGR to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to expected input size
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    # Normalize pixel values to 0-1
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Get predictions
    predictions = _model.predict(img_batch)[0]
    
    # Create dictionary of class probabilities
    result = {class_name: float(prob) for class_name, prob in zip(CLASSES, predictions)}
    
    return result

# Example usage
if __name__ == "__main__":
    # Example: Load an image and make a prediction
    sample_img_path = "dataset_resized/test/Cardiomegaly/00000032_053.png"  # Replace with actual path
    if os.path.exists(sample_img_path):
        img = cv2.imread(sample_img_path)
        result = predictor(img)
        print("Prediction results:")
        for condition, probability in result.items():
            print(f"{condition}: {probability:.4f}") 