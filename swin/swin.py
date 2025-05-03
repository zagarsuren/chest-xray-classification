#!/usr/bin/env python3
"""
swin.py: Provide a predictor function for Swin-B model inference.
Handles grayscale (e.g., X-ray) and RGB inputs by converting to RGB.
Example usage included at bottom.
"""

import os
import torch
from torchvision.models import swin_b
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Union

# Configuration: adjust model path
CHECKPOINT_PATH = "./swin/best_model/swinb_best.pth"
CLASS_NAMES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Nodule",
    "Pneumothorax"
]
IMG_SIZE = 224

# Load model once at import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = swin_b(weights=None, num_classes=len(CLASS_NAMES))
_model.to(device)
_ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
_model.load_state_dict(_ckpt, strict=False)
_model.eval()

# Preprocessing transform
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predictor(img: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
    """
    Run inference on an image (grayscale or RGB) using the Swin-B model.

    Args:
        img (np.ndarray or PIL.Image.Image): Input image as HxW or HxWxC numpy array or PIL Image.

    Returns:
        Dict[str, float]: Mapping from class names to probability scores.
    """
    # Convert numpy array to PIL Image and ensure RGB
    if isinstance(img, np.ndarray):
        arr = img.astype('uint8')
        # If single-channel or 2D grayscale
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode='L').convert('RGB')
        elif arr.ndim == 3 and arr.shape[2] == 1:
            pil_img = Image.fromarray(arr.squeeze(-1), mode='L').convert('RGB')
        # If BGR-ordered 3-channel from cv2, convert to RGB
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil_img = Image.fromarray(arr[..., ::-1], mode='RGB')
        else:
            raise ValueError(f"Unsupported numpy image shape {arr.shape}")
    elif isinstance(img, Image.Image):
        pil_img = img.convert('RGB')
    else:
        raise TypeError("Input must be a numpy array or PIL Image")

    # Apply transforms and prepare tensor
    input_tensor = _transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    return {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}


# Example usage
if __name__ == "__main__":
    import cv2

    sample_img_path = "./xray-swin/data_five_final/test/Effusion/00001376_007.png" 
    if os.path.exists(sample_img_path):
        img_bgr = cv2.imread(sample_img_path)
        if img_bgr is None:
            print(f"Failed to load image at {sample_img_path}")
        else:
            # predictor_swin expects RGB or grayscale; cv2.imread returns BGR
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = predictor(img_rgb)
            print("Prediction results:")
            for condition, probability in result.items():
                print(f"{condition}: {probability:.4f}")
    else:
        print(f"Image not found: {sample_img_path}")