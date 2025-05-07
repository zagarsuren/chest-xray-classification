import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------
# A) Directory Setup & Params
# -----------------------------
DATASET_PATH = r"C:\Users\arron\OneDrive\Documents\UTS\Post Graduate\Semester 4\49275 Neural Networks and Fuzzy Logic\Group Assignment\Datasets\1500_Xray\DATA\1500_train"
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Nodule', 'Pneumothorax']

TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VAL_PATH   = os.path.join(DATASET_PATH, 'val')
TEST_PATH  = os.path.join(DATASET_PATH, 'test')

IMG_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_CLASSES = len(CLASSES)
EPOCHS = 80

# -----------------------------
# B) Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15]
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# C) Model Construction (ResNet50)
# -----------------------------
CHECKPOINT_FILE = 'resnet50_best_model.h5'

if os.path.exists(CHECKPOINT_FILE):
    print("[INFO] Loading existing model...")
    model = tf.keras.models.load_model(CHECKPOINT_FILE)
else:
    print("[INFO] Building new ResNet50 model...")

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = True  # Allow fine-tuning

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=2e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# -----------------------------
# D) Callbacks
# -----------------------------
log_dir = os.path.join("logs_resnet50", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    TensorBoard(log_dir=log_dir),
    ModelCheckpoint(CHECKPOINT_FILE, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1)
]

# -----------------------------
# E) Training
# -----------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# -----------------------------
# F) Evaluation
# -----------------------------
loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

# Confusion Matrix
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_gen.class_indices.keys())
disp.plot(cmap=plt.cm.Greens)
plt.title("ResNet-50 Confusion Matrix")
plt.show()

# -----------------------------
# G) Classification Report
# -----------------------------
report = classification_report(y_true, y_pred, target_names=CLASSES)
print("\nTest Classification Report:\n")
print(report)

# Optional: save the report to a .txt file
with open("classification_report_resnet50.txt", "w") as f:
    f.write("Test Classification Report:\n\n")
    f.write(report)
