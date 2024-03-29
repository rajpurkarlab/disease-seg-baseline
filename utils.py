import numpy as np

def compute_segmentation_metrics(heatmap, gt_mask):
    best_iou = 0
    best_dice = 0
    best_thresh = 0
    
    for THRESHOLD in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = (heatmap > THRESHOLD).astype(int)

        intersection = np.logical_and(mask, gt_mask)
        union = np.logical_or(mask, gt_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        if iou_score > best_iou:
            best_iou = iou_score
            best_thresh = THRESHOLD
            best_dice = 2 * np.sum(intersection) / (np.sum(mask) + np.sum(gt_mask))

    return best_iou, best_dice, best_thresh

PATH_TO_ID = {
    # "Enlarged Cardiomediastinum": 0,
    "Cardiomegaly": 0,
    # "Lung Lesion": 2,
    # "Airspace Opacity": 3,
    "Edema": 1,
    "Consolidation": 2,
    "Atelectasis": 3,
    "Pneumothorax": 4,
    "Pleural Effusion": 5,
    # "Support Devices": 9
}

def read_prompts(filename):
    import pickle
    # Read dictionary pkl file
    with open(filename, 'rb') as fp:
        d = pickle.load(fp)
    return d