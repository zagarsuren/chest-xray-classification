import os
import torch
import numpy as np
from torchvision.models import swin_b, Swin_B_Weights
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def get_dataloaders(data_dir, batch_size=16):
    val_test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader, test_dataset.classes

def plot_roc(y_true, y_probs, class_names, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Class (Swin-B)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_swinb.png"))
    plt.close()
    print("Saved ROC curve to outputs/roc_curve_swinb.png")

def evaluate_model_auc(model, test_loader, device, class_names):
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    y_true_onehot = label_binarize(test_labels, classes=range(len(class_names)))

    # Compute per-class and macro AUC-ROC
    auc_per_class = roc_auc_score(y_true_onehot, test_probs, average=None)
    macro_auc = roc_auc_score(y_true_onehot, test_probs, average='macro')

    print(f"\nTest AUC-ROC (macro-average): {macro_auc:.4f}")
    for i, auc_val in enumerate(auc_per_class):
        print(f"AUC-ROC for class '{class_names[i]}': {auc_val:.4f}")

    report = classification_report(test_labels, test_preds, target_names=class_names)

    with open(os.path.join("outputs", "classification_report_swinb2.txt"), "w") as f:
        f.write(report)

    # Plot ROC curves for each class
    plot_roc(y_true_onehot, test_probs, class_names)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = "./xray-swin/data_five_final"
    model_path = "models/swinb_best.pth"
    batch_size = 16

    test_loader, class_names = get_dataloaders(data_dir, batch_size)

    weights = Swin_B_Weights.DEFAULT
    model = swin_b(weights=weights)
    model.head = torch.nn.Linear(model.head.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    evaluate_model_auc(model, test_loader, device, class_names)

if __name__ == "__main__":
    main()