import numpy as np
import tensorflow as tf
import cv2

# Load your trained ResNet50 model (adjust path if needed)
MODEL_PATH = 'resnet50_best_model.h5'

# List of your classes
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']

# Global model variable to avoid reloading
_model = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def predictor(img, model=None):
    """
    img: Numpy array (loaded by cv2)
    """
    model = model or load_model()

    # Preprocessing (resize, normalize)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]
    result = {class_name: float(prob) for class_name, prob in zip(CLASSES, preds)}
    return result
