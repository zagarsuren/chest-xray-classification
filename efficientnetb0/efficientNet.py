import os
# Set TF log level to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import cv2
import numpy as np

# Basic Settings
CLASSES = sorted(['Atelectasis','Cardiomegaly','Effusion', 'Nodule','Pneumothorax'])
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (512, 512)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
MODEL_VARIANT = 'EfficientNetB0'
DROPOUT_RATE = 0.3
best_checkpoint_path = "efficientnetb0/efficientnet.weights.h5"

def preprocess_image(image_cv2):
    img_resized = cv2.resize(image_cv2, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_rescaled = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_rescaled, axis=0)
    return img_batch

def build_keras_efficientnet(input_shape, num_classes, dropout_rate=0.2, model_name=""):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name="avg_pool"),
        layers.Dropout(dropout_rate, name="head_dropout"),
        layers.Dense(num_classes, activation='softmax', name="predictions")
    ], name=f"{model_name}_keras" if model_name else f"{MODEL_VARIANT}_keras")
    return model

def load_bestmodel(best_checkpoint_path):
    try:
        model_final = build_keras_efficientnet(INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, model_name="Final_Eval_Model")
        model_final.load_weights(best_checkpoint_path)
    except Exception as e:
        print(f"Warning: Could not load weights from {best_checkpoint_path}. Error: {e}")
    return model_final

def predictor(img):
    model = load_bestmodel(best_checkpoint_path)
    img_batch = preprocess_image(img)
    predictions = model.predict(img_batch)[0]
    result = {class_name: float(prob) for class_name, prob in zip(CLASSES, predictions)}
    return result

# Example usage
if __name__ == "__main__":
    sample_img_path = "/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/balanced_dataset_4/test/Cardiomegaly/00000032_053.png"
    if os.path.exists(sample_img_path):
        img = cv2.imread(sample_img_path)
        result = predictor(img)
        print("Prediction results:")
        for condition, probability in result.items():
            print(f"{condition}: {probability:.4f}") 


