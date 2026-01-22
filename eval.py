import os
import json
import sys
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from tabulate import tabulate
from torchvision.ops import masks_to_boxes, generalized_box_iou_loss
from monai.metrics import compute_hausdorff_distance

def compute_giou(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the Generalized Intersection over Union (gIoU) between predicted and ground truth masks.

    Args:
        pred (torch.Tensor): Predicted masks, shape (B, H, W).
        mask (torch.Tensor): Ground truth masks, shape (B, H, W). Values should be binary (0 or 1).

    Returns:
        torch.Tensor: gIoU for each element in the batch, shape (B,)
    """
    # Convert masks to bounding boxes
    # masks_to_boxes expects masks in (B, H, W) and returns (B, 4) with (x1, y1, x2, y2)
    pred_idx = pred.sum(dim=(1, 2)) != 0
    target_idx = mask.sum(dim=(1, 2)) != 0
    pred_boxes = masks_to_boxes(pred[pred_idx])  # Shape: (B, 4)
    target_boxes = masks_to_boxes(mask[target_idx])  # Shape: (B, 4)

    padded_pred_boxes = torch.zeros((pred.shape[0], 4), device=pred.device)
    padded_pred_boxes[pred_idx] = pred_boxes
    pred_boxes = padded_pred_boxes
    padded_target_boxes = torch.zeros((mask.shape[0], 4), device=pred.device)
    padded_target_boxes[target_idx] = target_boxes
    target_boxes = padded_target_boxes

    # Compute gIoU loss (which is 1 - gIoU)
    giou_loss = generalized_box_iou_loss(pred_boxes, target_boxes)  # Shape: (B,)

    # Convert loss to gIoU
    giou = 1 - giou_loss  # Shape: (B,)

    return giou

def score_cal(seg_map, prd_map):
    '''
    labels B * 1
    seg_map B * H * W
    prd_map B * H * W
    '''
    assert seg_map.ndim == prd_map.ndim
    assert seg_map.ndim >= 2
    if seg_map.ndim == 2:
        seg_map = seg_map.unsqueeze(0)
        prd_map = prd_map.unsqueeze(0)
        
    total_num = seg_map.shape[0]
    
    seg_map = seg_map.reshape(total_num, -1)
    prd_map = prd_map.reshape(total_num, -1)
    dot_product = (seg_map * prd_map)
    b_seg_map = 1 - seg_map
    b_prd_map = 1 - prd_map
    b_dot_product = (b_seg_map * b_prd_map)

    sum_dot = torch.sum(dot_product, dim=-1)
    sum_seg = torch.sum(seg_map, dim=-1)
    sum_prd = torch.sum(prd_map, dim=-1)
    b_sum_dot = torch.sum(b_dot_product, dim=-1)
    b_sum_seg = torch.sum(b_seg_map, dim=-1)
    b_sum_prd = torch.sum(b_prd_map, dim=-1)

    iou_score = sum_dot/((sum_seg + sum_prd)-sum_dot)
    dice_score = 2.*sum_dot / (sum_seg+sum_prd)
    
    b_iou_score = b_sum_dot/((b_sum_seg + b_sum_prd)-b_sum_dot)
    fb_iou_score = (iou_score + b_iou_score) / 2
    
    hd_score = compute_hausdorff_distance(prd_map.unsqueeze(1), seg_map.unsqueeze(1)).squeeze()

    return (iou_score, dice_score, fb_iou_score, hd_score)

def eval_seg(pred, mask):
    """
    Args:
        pred: [D,H,W]
        mask: [D,H,W]
    """
    (iou_score, dice_score, fb_iou_score, hd_score) = score_cal(mask, pred)
    
    iou_score[iou_score.isnan()] = 0. 
    dice_score[dice_score.isnan()] = 0.
    fb_iou_score[fb_iou_score.isnan()] = 0.
    hd_score[torch.logical_or(hd_score.isnan(), hd_score.isinf())] = 0.
    
    return (iou_score, dice_score, fb_iou_score, hd_score,)
    
if __name__ == "__main__":
    metrics = ["iou", "dice", "fb_iou", "hausdorf_dist",]
    dataset = sys.argv[1]
    print("=" * 100)
    print(f"Evaluating on dataset {dataset}")
    im_root = f"DATASETS/nnUnet_raw/{dataset}/imagesTs"
    pd_root = f"DATASETS/nnUnet_results/{dataset}/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict"
    gt_root = f"DATASETS/nnUnet_raw/{dataset}/labelsTs"
    json_path = f"DATASETS/nnUnet_raw/{dataset}/dataset.json"
    
    with open(json_path, "r") as f:
        categories = json.load(f)
    
    labels = {v: k for k, v in categories["labels"].items()}
    
    score_per_class = {
        name: {
            m: [] for m in metrics
        } for name in labels.values() if name != "background"
    }

    for case_ in tqdm(os.listdir(gt_root), desc=dataset, position=0):
        if os.path.splitext(case_)[-1] != ".gz":
            continue
        
        gt_path = os.path.join(gt_root, case_)
        pd_path = os.path.join(pd_root, case_)
        
        gt = nib.load(gt_path)
        pd = nib.load(pd_path)
        
        gt = gt.get_fdata()
        pd = pd.get_fdata()
        
        gt = torch.tensor(gt).permute(2, 0, 1).to(device="cuda:0")
        pd = torch.tensor(pd).permute(2, 0, 1).to(device="cuda:0")
        
        for cat in tqdm(labels.keys(), desc="Category", position=1, leave=False):
            if cat == 0:
                continue
            
            name = labels[cat]
            
            cat_gt = (gt == cat).to(torch.float32)
            cat_pd = (pd == cat).to(torch.float32)
            
            pos_slices = cat_gt.sum(dim=(1, 2)) > 0
            cat_gt = cat_gt[pos_slices]
            cat_pd = cat_pd[pos_slices]
            
            if cat_gt.numel() == 0:
                continue
            
            (iou_score, dice_score, fb_iou_score, hd_score,) = eval_seg(cat_pd, cat_gt)
            
            score_per_class[name]["iou"].extend(iou_score.tolist())
            score_per_class[name]["dice"].extend(dice_score.tolist())
            score_per_class[name]["fb_iou"].extend(fb_iou_score.tolist())
            score_per_class[name]["hausdorf_dist"].extend(hd_score.tolist())

    score_per_class["Avg"] = { m: [] for m in metrics }
    for name, metrics_dict in score_per_class.items():
        if name == "Avg":
            continue
        mean_metrics = {k: np.mean(v) for k, v in metrics_dict.items()}
        score_per_class[name] = mean_metrics
        for metric in score_per_class["Avg"].keys():
            score_per_class["Avg"][metric].append(score_per_class[name][metric])
            
    score_per_class["Avg"] = {k: np.mean(v) for k, v in score_per_class["Avg"].items()}
    
    tabulate_results = []
    for name, metrics_dict in score_per_class.items():
        tabulate_results.append((name,) + tuple(metrics_dict.values()))
        
    print(tabulate(tabulate_results, headers=["class"] + metrics, tablefmt="grid", floatfmt=".4f"))