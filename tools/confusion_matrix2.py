import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_confusion_matrix(predicted_folder, target_folder):
    predicted_files = os.listdir(predicted_folder)
    target_files = os.listdir(target_folder)

    if len(predicted_files) != len(target_files):
        raise ValueError("Number of predicted files and target files do not match.")

    predicted_labels = []
    target_labels = []

    for file in predicted_files:
        predicted_labels.append(os.path.splitext(file)[0])
    for file in target_files:
        target_labels.append(os.path.splitext(file)[0])

    labels = np.unique(predicted_labels + target_labels)

    predicted_labels = np.array([np.where(labels == label)[0][0] for label in predicted_labels])
    target_labels = np.array([np.where(labels == label)[0][0] for label in target_labels])

    cm = confusion_matrix(target_labels, predicted_labels)

    return cm

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python confusion_matrix.py predicted_folder target_folder")
        sys.exit(1)

    predicted_folder = sys.argv[1]
    target_folder = sys.argv[2]

    confusion_matrix = calculate_confusion_matrix(predicted_folder, target_folder)
    print(confusion_matrix)
