import os
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from scipy.stats import chi2
import torch
from pathlib import Path

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    pred_label = torch.from_numpy((pred_label))
    label = torch.from_numpy(label)

    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id

    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def evaluate(pred_folder, ground_truth, pred_prefix, gt_prefix, num_classes=2):
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)

    for file in os.listdir(ground_truth):
        if file.endswith(gt_prefix):
            filename = Path(file).stem
            gt_path = os.path.join(ground_truth, filename + gt_prefix)
            pred_path = os.path.join(pred_folder, filename + pred_prefix)

            target_label = np.array(Image.open(gt_path))
            pred = np.array(Image.open(pred_path))    
    
            area_intersect, area_union, area_pred_label, area_label = \
                intersect_and_union(pred, target_label, num_classes)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
    return chi2, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Comparison Script")
    parser.add_argument("pred_folder", help="Path to prediction folder 1")
    parser.add_argument("ground_truth", help="Path to ground truth folder")
    parser.add_argument("--pred_prefix", default='.png',help="Path to ground truth folder")
    parser.add_argument("--gt_prefix", default='.jpg',help="Path to ground truth folder")
    args = parser.parse_args()

    pred_folder = args.pred_folder
    ground_truth = args.ground_truth
    pred_prefix = args.pred_prefix
    gt_prefix = args.gt_prefix

    result = evaluate(pred_folder, ground_truth, pred_prefix, gt_prefix)
