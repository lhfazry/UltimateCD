import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(target_folder, predicted_folder):
    # Load the ground-truth and predicted images
    target_images = load_images_from_folder(target_folder)
    predicted_images = load_images_from_folder(predicted_folder)

    # Flatten the images
    target_flat = np.concatenate(target_images).ravel()
    predicted_flat = np.concatenate(predicted_images).ravel()

    # Compute the confusion matrix
    cm = confusion_matrix(target_flat, predicted_flat)

    return cm

def load_images_from_folder(folder):
    # Load binary images from the specified folder
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute confusion matrix for image segmentation')
    parser.add_argument('target_folder', help='Folder containing ground-truth binary images')
    parser.add_argument('predicted_folder', help='Folder containing segmentation result images')
    args = parser.parse_args()

    # Compute the confusion matrix
    cm = compute_confusion_matrix(args.target_folder, args.predicted_folder)

    # Print the confusion matrix
    print('Confusion Matrix:')
    print(cm)
