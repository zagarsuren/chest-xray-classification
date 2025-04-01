import os
import torch.multiprocessing as mp

# Set sharing strategy to file_system to avoid shared memory issues
mp.set_sharing_strategy('file_system')

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    SwinForImageClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# 1. Load dataset
data_dir = "/Users/zagarsuren/Documents/GitHub/xray-classification/data"  # path to the folder containing train/val/test splits
dataset = load_dataset("imagefolder", data_dir=data_dir)
print("Dataset splits:", dataset.keys())
labels = dataset["train"].features["label"].names
num_classes = len(labels)

# 2. Load image processor and model
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
torch.device("cpu")
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True # Ignore classifier size mismatch
)

device = torch.device("cpu")
model.to(device)

# 3. Transform function
def transform(examples):
    # Process images: convert to RGB if needed.
    processed_images = []
    for image in examples["image"]:
        # If the image is a PIL image, check its mode
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        # If the image is a numpy array with only 2 dimensions, add a channel dimension and repeat it across channels.
        elif isinstance(image, np.ndarray) and image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        processed_images.append(image)
    
    # Process the images with the image processor
    inputs = image_processor(processed_images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs


train_ds = dataset["train"].with_transform(transform)
val_ds   = dataset["validation"].with_transform(transform)
test_ds  = dataset["test"].with_transform(transform)

# 4. Collate function
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["labels"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# 5. Metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 6. Training arguments with TensorBoard reporting
training_args = TrainingArguments(
    output_dir="swin-chestxray-output",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="tensorboard",  # Log to TensorBoard
    logging_dir="runs",       # TensorBoard log directory
    dataloader_num_workers=0  # Disable multiprocessing in DataLoader
)


# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# 8. Train the model
trainer.train()

# 9. Evaluate on test set
test_metrics = trainer.evaluate(test_ds)
print("Test metrics:", test_metrics)

# 10. Save the best model
best_model_dir = "swin-chestxray-output/best_model"
print("Saving best model to", best_model_dir)
trainer.save_model(best_model_dir)
image_processor.save_pretrained(best_model_dir)
print("Best model and image processor saved successfully.")
