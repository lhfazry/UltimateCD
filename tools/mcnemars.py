import os
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from scipy.stats import chi2
import torch
#from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

def intersect_and_union(pred1,
                        pred2,
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
    pred1 = torch.from_numpy(np.load(pred1))
    pred2 = torch.from_numpy(np.load(pred2))
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
    pred1 = pred1[mask]
    pred2 = pred2[mask]
    label = label[mask]

    n00 = pred1[((pred1==label)==0) & ((pred2==label)==0)] # both misclassified
    n01 = pred1[((pred1==label)==0) & ((pred2==label)==1)] # misclassified only by pred1
    n10 = pred1[((pred1==label)==1) & ((pred2==label)==0)] # misclassified only by pred2
    n11 = pred1[((pred1==label)==1) & ((pred2==label)==1)] # both correct

    area_n00 = torch.histc(n00.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_n01 = torch.histc(n01.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_n10 = torch.histc(n10.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_n11 = torch.histc(n11.float(), bins=(num_classes), min=0, max=num_classes - 1)

    return area_n00, area_n01, area_n10, area_n11


def perform_mcnemar_test(pred_folder1, pred_folder2, ground_truth):
    contingency_table = np.zeros((2, 2))

    for file in os.listdir(ground_truth):
        #if file.endswith(".png"):
        gt_path = os.path.join(ground_truth, file)
        pred_path1 = os.path.join(pred_folder1, file)
        pred_path2 = os.path.join(pred_folder2, file)

        print(gt_path, pred_path1, pred_path2)
        target_label = np.array(Image.open(gt_path))# // 255
        print(target_label)
        pred1 = np.array(Image.open(pred_path1)) // 255
        pred2 = np.array(Image.open(pred_path2)) // 255

        ct = mcnemar_table(y_target=target_label.flatten(), 
                y_model1=pred1.flatten(), 
                y_model2=pred2.flatten())
        
        contingency_table = contingency_table + ct
        
    chi2, p = mcnemar(ary=contingency_table)
    
    return chi2, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Comparison Script")
    parser.add_argument("pred_folder1", help="Path to prediction folder 1")
    parser.add_argument("pred_folder2", help="Path to prediction folder 2")
    parser.add_argument("ground_truth", help="Path to ground truth folder")
    args = parser.parse_args()

    pred_folder1 = args.pred_folder1
    pred_folder2 = args.pred_folder2
    ground_truth = args.ground_truth

    chi2, p = perform_mcnemar_test(pred_folder1, pred_folder2, ground_truth)
    print("chi2:", chi2)
    print("p:", p)
