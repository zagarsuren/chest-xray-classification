import cv2
import os
from densenet121.densenet import predictor

sample_img_path = "/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/balanced_dataset_4/test/Cardiomegaly/00000032_053.png"  # Replace with actual path
if os.path.exists(sample_img_path):
    img = cv2.imread(sample_img_path)
    result = predictor(img)
    print("Prediction results:")
    for condition, probability in result.items():
        print(f"{condition}: {probability:.4f}") 