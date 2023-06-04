import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

def calculate_confusion_matrix(predicted_folder, target_folder):
    predicted_files = os.listdir(predicted_folder)
    target_files = os.listdir(target_folder)

    if len(predicted_files) != len(target_files):
        raise ValueError("Number of predicted files and target files do not match.")

    predicted_labels = []
    target_labels = []

    for file in predicted_files:
        image_path = os.path.join(predicted_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        predicted_labels.append(image.flatten())

    for file in target_files:
        image_path = os.path.join(target_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        target_labels.append(image.flatten())

    predicted_labels = np.array(predicted_labels)
    target_labels = np.array(target_labels)

    all_labels = np.unique(np.concatenate((predicted_labels, target_labels)))
    label_mapping = {label: i for i, label in enumerate(all_labels)}

    predicted_labels = np.array([label_mapping[label] for label in predicted_labels])
    target_labels = np.array([label_mapping[label] for label in target_labels])

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