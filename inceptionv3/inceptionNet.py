import os
# Set TF log level before import to reduce verbosity (optional: 0, 1, 2, 3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3 # Using InceptionV3
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import sys

# -----------------------------
# A) Configuration & Setup
# -----------------------------
print(f"Using TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# --- Basic Settings ---
DATA_ROOT = '/home/sagemaker-user/AUT2025/X_ray/DATA/1500_train' # Example path - ADJUST IF NEEDED
NUM_CLASSES = 5
CLASSES = sorted(['Atelectasis', 'Cardiomegaly', 'Nodule', 'Pneumothorax', 'No Finding']) # Make sure these match your folders

# --- Model & Image Settings ---
MODEL_VARIANT = 'InceptionV3'
IMG_SIZE = (512, 512) # Standard input size for InceptionV3
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)

# --- Training Hyperparameters (Single Phase - Full Fine-Tuning) ---
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 300      # Max epochs (ES will likely stop sooner)
# Note: Using SGD with LR=1e-4 based on user script. Monitor for stability.
# Consider Adam optimizer for InceptionV3, potentially starting with a slightly higher LR like 1e-4 or 5e-5.
LEARNING_RATE = 1e-4

# --- Regularization ---
DROPOUT_RATE = 0.3      # Adjust based on overfitting/underfitting (e.g., 0.2-0.5)
WEIGHT_DECAY = 1e-4     # Adjust if needed (e.g., 1e-5, 5e-4) - Applicable mainly to AdamW or TFW optimizers. SGD handles it via kernel_regularizer if needed, or directly in optimizer args in TF > 2.x

# --- Callbacks ---
LR_PATIENCE = 10        # Patience for LR reduction
LR_FACTOR = 0.2
ES_MONITOR = 'val_loss'
ES_PATIENCE = 25        # Patience for early stopping
ES_MODE = 'min'
CKPT_MONITOR = 'val_loss'
CKPT_MODE = 'min'

# --- Directories ---
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
LOG_BASE_DIR = 'logs_keras_simple' # Base directory for logs
CHECKPOINT_BASE_DIR = 'checkpoints_keras_simple' # Base directory for checkpoints

os.makedirs(LOG_BASE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# -----------------------------
# Pre-processing before prediction
# -----------------------------
def preprocess_image(image_cv2):
    # 1. Resize the image to the target size (IMG_SIZE) used during training
    img_resized = cv2.resize(image_cv2, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Rescale pixel values to the range [0, 1]
    img_rescaled = img_resized.astype(np.float32) / 255.0

    # 3. Add a batch dimension at the beginning.
    img_batch = np.expand_dims(img_rescaled, axis=0)

    return img_batch


# -----------------------------
# Model Building Function
# -----------------------------
def build_keras_inceptionv3(input_shape, num_classes, dropout_rate=0.2, model_name=""):
    """Builds an InceptionV3 model with ALL base layers unfrozen."""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True # Unfreeze the entire base model
    print(f"Base model ({MODEL_VARIANT}) loaded. All layers are set to trainable.")

    # Add L2 regularization to Dense layers if needed (via kernel_regularizer)
    # from tensorflow.keras import regularizers
    # kernel_reg = regularizers.l2(WEIGHT_DECAY) # If using L2

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name="avg_pool"),
        layers.Dropout(dropout_rate, name="head_dropout"),
        layers.Dense(num_classes, activation='softmax', name="predictions"
                     # kernel_regularizer=kernel_reg # Example L2
                     )
    ], name=f"{model_name}_keras" if model_name else f"{MODEL_VARIANT}_keras")
    return model


# -----------------------------
# Load Final Best Weights and Evaluate
# -----------------------------
print(f"\n--- Loading Final Best Weights and Evaluating ---")
best_checkpoint_path = "../keras_inceptionNet_5class/checkpoints_keras_simple/InceptionV3_full_finetune_20250427-122319/InceptionV3_best_weights.weights.h5"

def load_bestmodel(best_checkpoint_path):
    print(f"Loading best weights from: {best_checkpoint_path}")
    try:
        model_final = build_keras_inceptionv3(INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, model_name="Final_Eval_Model")
        model_final.load_weights(best_checkpoint_path)
        print("Successfully loaded best weights.")
    except Exception as e:
        print(f"Warning: Could not load weights from {best_checkpoint_path}. "
              f"Evaluation might use randomly initialized weights or weights from end of training if EarlyStopping didn't restore. Error: {e}")
    
    return model_final



def predictor(img):
    """
    Predict chest X-ray conditions from an OpenCV image.
    
    Args:
        img: Numpy array containing the image (BGR format from OpenCV)
        
    Returns:
        Dictionary mapping class names to confidence scores
    """
    model = load_bestmodel(best_checkpoint_path)
    img_batch = preprocess_image(img)
    # Get predictions
    predictions = model.predict(img_batch)[0]
    
    # Create dictionary of class probabilities
    result = {class_name: float(prob) for class_name, prob in zip(CLASSES, predictions)}
    
    return result

# Example usage
if __name__ == "__main__":
    # Example: Load an image and make a prediction
    sample_img_path = "../DATA/1500_train/test/Cardiomegaly/00004533_011.png"  # Replace with actual path
    if os.path.exists(sample_img_path):
        img = cv2.imread(sample_img_path)
        result = predictor(img)
        print("Prediction results:")
        for condition, probability in result.items():
            print(f"{condition}: {probability:.4f}") 