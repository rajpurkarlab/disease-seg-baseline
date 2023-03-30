import json
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from uuid import uuid4
import pycocotools.mask as mask_util
import argparse

from models.run_biovil import plot_phrase_grounding as ppgb
from models.BioViL.image.data.io import load_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of model (BioViL, )')
    parser.add_argument('test_set', type=str, help='name of test set (CheXlocalize, )')
    parser.add_argument('visualize', type=str, help='yes or no')
    return parser.parse_args()

PROMPTS = {
    "Enlarged Cardiomediastinum": "Findings suggesting enlarged cardiomediastinum",
    "Cardiomegaly": "Findings suggesting cardiomegaly", 
    "Lung Lesion": "Fundings suggesting lung lesions", 
    "Airspace Opacity": "Findings suggesting airspace opacities",
    "Edema": "Findings suggesting an edema",
    "Consolidation": "Findings suggesting consolidation",
    "Atelectasis": "Findings suggesting atelectasis",
    "Pneumothorax": "Findings suggesting a pneumothorax",
    "Pleural Effusion": "Findings suggesting pleural effusion", 
    "Support Devices": "Findings suggesting support devices" # TODO: change this?
}

PATH_TO_ID = {
    "Enlarged Cardiomediastinum": 0,
    "Cardiomegaly": 1,
    "Lung Lesion": 2,
    "Airspace Opacity": 3,
    "Edema": 4,
    "Consolidation": 5,
    "Atelectasis": 6,
    "Pneumothorax": 7,
    "Pleural Effusion": 8,
    "Support Devices": 9
}

def main():
    args = parse_args()

    PLOT_IMAGES = False
    if args.visualize == "yes":
        PLOT_IMAGES = True

    if args.model != "BioViL":
        raise NotImplementedError("Only BioViL is implemented for now")
    if args.test_set != "CheXlocalize":
        raise NotImplementedError("Only CheXlocalize is implemented for now")
    
    ious_by_pathology = [0]*len(PROMPTS)
    dices_by_pathology = [0]*len(PROMPTS)
    numbers_by_pathology = [0]*len(PROMPTS)

    json_obj = json.load(open("datasets/CheXlocalize/gt_segmentations_test.json"))

    for obj in json_obj:
        filename = "datasets/CheXlocalize/CheXpert/test/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        for query in json_obj[obj]:
            text_prompt = PROMPTS[query]
            heatmap = ppgb(filename, text_prompt)
            annots = json_obj[obj][query]

            if annots['counts'] != 'ifdl3':
                gt_mask = mask_util.decode(annots)
                
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

                    intersection = np.logical_and(mask, gt_mask)
                    dice_score = 2 * np.sum(intersection) / (np.sum(mask) + np.sum(gt_mask))
                    best_dice = max(best_dice, dice_score)
                
                if PLOT_IMAGES:
                    _, axes = plt.subplots(1, 3, figsize=(15, 6))
                    image = load_image(Path(filename)).convert("RGB")
                    axes[0].imshow(image)
                    axes[0].axis('off')
                    axes[0].set_title("Input image")

                    axes[1].imshow((heatmap > best_thresh).astype(int))
                    axes[1].axis('off')
                    axes[1].set_title(f"BioViL mask: {text_prompt}")

                    axes[2].imshow(gt_mask)
                    axes[2].axis('off')
                    axes[2].set_title(f"GT mask: {query}")
                    plt.savefig(f"biovil_plot_{uuid4()}.png")
                
                ious_by_pathology[PATH_TO_ID[query]] += best_iou
                dices_by_pathology[PATH_TO_ID[query]] += best_dice
                numbers_by_pathology[PATH_TO_ID[query]] += 1
    
    for pathology in PROMPTS:
        print("Pathology:", pathology, "mIoU:", ious_by_pathology[PATH_TO_ID[pathology]]/numbers_by_pathology[PATH_TO_ID[pathology]], "Avg. DICE:", dices_by_pathology[PATH_TO_ID[pathology]]/numbers_by_pathology[PATH_TO_ID[pathology]])

    print("mIoU:", sum(ious_by_pathology)/sum(numbers_by_pathology))
    print("Avg. DICE: ", sum(dices_by_pathology)/sum(numbers_by_pathology))

if __name__ == "__main__":
    main()