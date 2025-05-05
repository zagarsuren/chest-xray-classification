import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report
from ensemble_model_inference import EnsembleModelClassifier

import tracemalloc, linecache


def evaluate_ensemble_classifier(test_dir, batch_size=10, results_dir="evaluation_results"):
    """
    Evaluates the ensemble classifier on a test dataset with memory efficiency.
    
    Args:
        test_dir (str): Path to the test directory containing class subdirectories.
        batch_size (int): Number of images to process before updating metrics.
        results_dir (str): Directory to save intermediate results.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Initialize the ensemble classifier
    ensemble = EnsembleModelClassifier()
    classes = ensemble.classes
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Files to store results
    predictions_file = os.path.join(results_dir, "predictions.csv")
    conf_matrix_file = os.path.join(results_dir, "confusion_matrix.csv")
    metrics_file = os.path.join(results_dir, "metrics.txt")
    
    # Initialize confusion matrix
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Initialize per-class metrics tracking
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    true_negatives = np.zeros(num_classes)
    
    # Initialize tracking variables
    total_processed = 0
    correct_predictions = 0
    
    # Initialize or open predictions file
    if os.path.exists(predictions_file):
        predictions_mode = 'a'  # Append mode if file exists
    else:
        predictions_mode = 'w'  # Create new file if it doesn't exist
    
    # Initialize or load confusion matrix
    if os.path.exists(conf_matrix_file):
        try:
            # Load existing confusion matrix from CSV
            with open(conf_matrix_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for i, row in enumerate(reader):
                    for j, val in enumerate(row[1:]):  # Skip row label
                        conf_matrix[i, j] = int(val)
            
            # Recalculate class metrics from confusion matrix
            for i in range(num_classes):
                true_positives[i] = conf_matrix[i, i]
                false_positives[i] = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
                false_negatives[i] = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
                true_negatives[i] = np.sum(conf_matrix) - true_positives[i] - false_positives[i] - false_negatives[i]
            
            total_processed = np.sum(conf_matrix)
            correct_predictions = np.sum(np.diag(conf_matrix))
            
            print(f"Loaded existing confusion matrix with {total_processed} predictions")
        except Exception as e:
            print(f"Error loading confusion matrix: {e}")
    
    # Save initial confusion matrix
    save_confusion_matrix(conf_matrix, classes, conf_matrix_file)
    
    # Open predictions file to record results
    with open(predictions_file, predictions_mode) as f_pred:
        # Write header if new file
        if predictions_mode == 'w':
            f_pred.write("image_path,true_class,true_class_idx,predicted_class,predicted_class_idx\n")
        
        # Get all image paths organized by class
        print(f"Scanning test directory: {test_dir}")
        class_paths = {}
        for class_name in classes:
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory for class {class_name} not found at {class_dir}")
                continue
                
            class_paths[class_name] = []
            for filename in os.listdir(class_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    class_paths[class_name].append(img_path)
        
        # Calculate total number of images
        total_images = sum(len(paths) for paths in class_paths.values())
        print(f"Found {total_images} images across {len(class_paths)} classes")
        
        # Process each class one at a time
        for class_name, img_paths in class_paths.items():
            class_idx = classes.index(class_name)
            print(f"Processing class: {class_name} ({len(img_paths)} images)")
            
            # Process images in batches to save memory
            for i in range(0, len(img_paths), batch_size):
                batch_paths = img_paths[i:i+batch_size]
                print(f"  Processing batch {i//batch_size + 1}/{(len(img_paths)-1)//batch_size + 1} "
                      f"({i+1}-{min(i+batch_size, len(img_paths))}/{len(img_paths)})")
                
                # Process each image in the batch
                for img_path in batch_paths:
                    total_processed += 1
                    
                    # Get prediction from ensemble model
                    try:
                        weighted_scores, predicted_class = ensemble.predict(img_path)
                        
                        if not weighted_scores:
                            print(f"Warning: Failed to get prediction for {img_path}")
                            continue
                            
                        predicted_idx = classes.index(predicted_class)
                        
                        # Update confusion matrix
                        conf_matrix[class_idx][predicted_idx] += 1
                        
                        # Update TP, FP, FN, TN for each class
                        for c in range(num_classes):
                            if c == class_idx and c == predicted_idx:
                                # True Positive for this class
                                true_positives[c] += 1
                            elif c != class_idx and c == predicted_idx:
                                # False Positive for this class
                                false_positives[c] += 1
                            elif c == class_idx and c != predicted_idx:
                                # False Negative for this class
                                false_negatives[c] += 1
                            else:
                                # True Negative for this class
                                true_negatives[c] += 1
                        
                        # Track correct predictions
                        if predicted_idx == class_idx:
                            correct_predictions += 1
                        
                        # Write result to predictions file
                        f_pred.write(f"{img_path},{class_name},{class_idx},{predicted_class},{predicted_idx}\n")
                        f_pred.flush()  # Ensure data is written immediately
                        
                        # Save confusion matrix after each prediction
                        save_confusion_matrix(conf_matrix, classes, conf_matrix_file)
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
                
                # Calculate and print current metrics
                precision, recall, f1, accuracy = calculate_metrics(true_positives, false_positives, 
                                                                    false_negatives, correct_predictions, 
                                                                    total_processed)
                
                # Print progress after each batch
                if total_processed > 0:
                    print(f"  Current accuracy: {accuracy:.4f} ({correct_predictions}/{total_processed})")
                    save_current_metrics(precision, recall, f1, accuracy, classes, metrics_file)
            
            # Print progress after each class
            if total_processed > 0:
                precision, recall, f1, accuracy = calculate_metrics(true_positives, false_positives, 
                                                                    false_negatives, correct_predictions, 
                                                                    total_processed)
                print(f"Processed {total_processed}/{total_images} images. Current accuracy: {accuracy:.4f}")
    
    # Calculate final metrics
    final_metrics = {}
    if total_processed > 0:
        precision, recall, f1, accuracy = calculate_metrics(true_positives, false_positives, 
                                                          false_negatives, correct_predictions, 
                                                          total_processed)
        
        # Prepare results dictionary
        final_metrics = {
            "precision": {classes[i]: precision[i] for i in range(len(classes))},
            "recall": {classes[i]: recall[i] for i in range(len(classes))},
            "f1_score": {classes[i]: f1[i] for i in range(len(classes))},
            "macro_avg": {
                "precision": np.mean(precision),
                "recall": np.mean(recall),
                "f1_score": np.mean(f1)
            },
            "weighted_avg": {
                "precision": np.sum(precision * np.sum(conf_matrix, axis=1)) / np.sum(conf_matrix),
                "recall": np.sum(recall * np.sum(conf_matrix, axis=1)) / np.sum(conf_matrix),
                "f1_score": np.sum(f1 * np.sum(conf_matrix, axis=1)) / np.sum(conf_matrix)
            },
            "average_accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "normalized_confusion_matrix": conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10),
        }
    
    return final_metrics

def calculate_metrics(tp, fp, fn, correct_predictions, total_processed):
    """
    Calculate precision, recall, F1 score, and accuracy from confusion matrix values.
    
    Args:
        tp (numpy.ndarray): True positives for each class
        fp (numpy.ndarray): False positives for each class
        fn (numpy.ndarray): False negatives for each class
        correct_predictions (int): Total correct predictions
        total_processed (int): Total processed samples
        
    Returns:
        tuple: (precision, recall, f1, accuracy) as numpy arrays and float
    """
    # Calculate precision and recall
    precision = np.zeros_like(tp, dtype=float)
    recall = np.zeros_like(tp, dtype=float)
    f1 = np.zeros_like(tp, dtype=float)
    
    # Calculate metrics for each class
    for i in range(len(tp)):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    # Overall accuracy
    accuracy = correct_predictions / total_processed if total_processed > 0 else 0
    
    return precision, recall, f1, accuracy

def save_confusion_matrix(conf_matrix, class_names, filename):
    """
    Save the confusion matrix to a CSV file.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix to save
        class_names (list): List of class names
        filename (str): Path to save the CSV file
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header with class names
        writer.writerow([''] + class_names)
        # Write each row with class name as first column
        for i, class_name in enumerate(class_names):
            writer.writerow([class_name] + conf_matrix[i].tolist())

def save_current_metrics(precision, recall, f1, accuracy, class_names, filename):
    """
    Save current metrics to a text file.
    
    Args:
        precision (numpy.ndarray): Precision values for each class
        recall (numpy.ndarray): Recall values for each class
        f1 (numpy.ndarray): F1 scores for each class
        accuracy (float): Overall accuracy
        class_names (list): List of class names
        filename (str): Path to save the metrics file
    """
    with open(filename, 'w') as f:
        f.write("===== CURRENT METRICS =====\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-----------------\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall: {recall[i]:.4f}\n")
            f.write(f"  F1-Score: {f1[i]:.4f}\n")
        
        f.write("\nAverage Metrics:\n")
        f.write("---------------\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n\n")
        
        f.write("  Macro Average:\n")
        f.write(f"    Precision: {np.mean(precision):.4f}\n")
        f.write(f"    Recall: {np.mean(recall):.4f}\n")
        f.write(f"    F1-Score: {np.mean(f1):.4f}\n")

def plot_confusion_matrices(results, class_names, results_dir="evaluation_results"):
    """
    Plots and saves the confusion matrices (regular and normalized).
    
    Args:
        results (dict): Dictionary containing the evaluation results.
        class_names (list): List of class names.
        results_dir (str): Directory to save the plots.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot regular confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()  # Close to free memory
    print(f"Confusion matrix saved to {os.path.join(results_dir, 'confusion_matrix.png')}")
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results["normalized_confusion_matrix"], annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "normalized_confusion_matrix.png"))
    plt.close()  # Close to free memory
    print(f"Normalized confusion matrix saved to {os.path.join(results_dir, 'normalized_confusion_matrix.png')}")

def save_metrics_report(results, class_names, results_dir="evaluation_results", output_file="evaluation_report.txt"):
    """
    Saves a detailed metrics report to a text file.
    
    Args:
        results (dict): Dictionary containing the evaluation results.
        class_names (list): List of class names.
        results_dir (str): Directory to save the report.
        output_file (str): Filename to save the report.
    """
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, output_file)
    
    with open(report_path, 'w') as f:
        f.write("===== CHEST X-RAY ENSEMBLE CLASSIFIER EVALUATION =====\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-----------------\n")
        for class_name in class_names:
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {results['precision'][class_name]:.4f}\n")
            f.write(f"  Recall: {results['recall'][class_name]:.4f}\n")
            f.write(f"  F1-Score: {results['f1_score'][class_name]:.4f}\n")
        
        f.write("\nAverage Metrics:\n")
        f.write("---------------\n")
        f.write(f"  Accuracy: {results['average_accuracy']:.4f}\n\n")
        
        f.write("  Macro Average:\n")
        f.write(f"    Precision: {results['macro_avg']['precision']:.4f}\n")
        f.write(f"    Recall: {results['macro_avg']['recall']:.4f}\n")
        f.write(f"    F1-Score: {results['macro_avg']['f1_score']:.4f}\n\n")
        
        f.write("  Weighted Average:\n")
        f.write(f"    Precision: {results['weighted_avg']['precision']:.4f}\n")
        f.write(f"    Recall: {results['weighted_avg']['recall']:.4f}\n")
        f.write(f"    F1-Score: {results['weighted_avg']['f1_score']:.4f}\n")
    
    print(f"Detailed evaluation report saved to {report_path}")

if __name__ == "__main__":
    tracemalloc.start()
    
    test_directory = "/home/diego/Documents/master/S4/Fuzzy_Logic/DenseNet121-Chest-X-Ray/Balanced_5_classes/test"
    results_directory = "evaluation_results"
    
    print("Starting evaluation of ensemble classifier...")
    # Use a batch size of 2 for testing, adjust as needed
    metrics = evaluate_ensemble_classifier(test_directory, batch_size=2, results_dir=results_directory)
    
    if metrics:
        # Get class names from the first ensemble classifier instance
        ensemble = EnsembleModelClassifier()
        class_names = ensemble.classes
        
        # Print metrics
        print("\n===== EVALUATION RESULTS =====")
        
        print("\nPrecision for each class:")
        for class_name, value in metrics["precision"].items():
            print(f"  {class_name}: {value:.4f}")
            
        print("\nRecall for each class:")
        for class_name, value in metrics["recall"].items():
            print(f"  {class_name}: {value:.4f}")
            
        print("\nF1-Score for each class:")
        for class_name, value in metrics["f1_score"].items():
            print(f"  {class_name}: {value:.4f}")
        
        print("\nMacro Average Metrics:")
        print(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
        print(f"  Recall: {metrics['macro_avg']['recall']:.4f}")
        print(f"  F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
        
        print("\nWeighted Average Metrics:")
        print(f"  Precision: {metrics['weighted_avg']['precision']:.4f}")
        print(f"  Recall: {metrics['weighted_avg']['recall']:.4f}")
        print(f"  F1-Score: {metrics['weighted_avg']['f1_score']:.4f}")
            
        print(f"\nAverage Accuracy: {metrics['average_accuracy']:.4f}")
        
        # Plot confusion matrices
        plot_confusion_matrices(metrics, class_names, results_directory)
        
        # Save detailed report
        save_metrics_report(metrics, class_names, results_directory)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        for stat in top_stats[:10]:
            frame = stat.traceback[0]
            code  = linecache.getline(frame.filename, frame.lineno).strip()
            print(f"{frame.filename}:{frame.lineno} "
                f"{stat.size/1024:.1f} KiB | {code}")
    else:
        print("Evaluation failed.") 