import json
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from uuid import uuid4
import pycocotools.mask as mask_util

from models.run_biovil import plot_phrase_grounding as ppgb
from models.BioViL.image.data.io import load_image

PROMPTS = {
    "Enlarged Cardiomediastinum": "There is an enlarged cardiomediastinum",
    "Cardiomegaly": "There is a cardiomegaly", 
    "Lung Lesion": "There is a lung lesion", 
    "Airspace Opacity": "There is an airspace opacity",
    "Edema": "There is an edema",
    "Consolidation": "There is consolidation",
    "Atelectasis": "There is atelectasis",
    "Pneumothorax": "There is a pneumothorax",
    "Pleural Effusion": "There is pleural effusion", 
    "Support Devices": "There are support devices"
}

def main():
    PLOT_IMAGES = False

    ious = []
    dices = []

    json_obj = json.load(open("datasets/CheXlocalize/gt_segmentations_test.json"))

    for obj in json_obj:
        filename = "datasets/CheXlocalize/CheXpert/test/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        for query in json_obj[obj]:
            text_prompt = PROMPTS[query]
            heatmap = ppgb(filename, text_prompt)
            # THRESHOLD = np.nanmin(heatmap)+np.nanmax(heatmap)
            
            annots = json_obj[obj][query]

            if annots['counts'] != 'ifdl3':
                gt_mask = mask_util.decode(annots)

                if PLOT_IMAGES:
                    _, axes = plt.subplots(1, 3, figsize=(15, 6))
                    image = load_image(Path(filename)).convert("RGB")
                    axes[0].imshow(image)
                    axes[0].axis('off')
                    axes[0].set_title("Input image")

                    axes[1].imshow(mask)
                    axes[1].axis('off')
                    axes[1].set_title(f"BioViL mask: {text_prompt}")

                    axes[2].imshow(gt_mask)
                    axes[2].axis('off')
                    axes[2].set_title(f"GT mask: {query}")
                    plt.savefig(f"biovil_plot_{uuid4()}.png")
                
                best_iou = 0
                best_dice = 0
                
                for THRESHOLD in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    mask = (heatmap > THRESHOLD).astype(int)

                    # compute iou with gt_mask using own code
                    intersection = np.logical_and(mask, gt_mask)
                    union = np.logical_or(mask, gt_mask)
                    iou_score = np.sum(intersection) / np.sum(union)
                    best_iou = max(best_iou, iou_score)

                    #compute dice score with gt_mask using own code
                    intersection = np.logical_and(mask, gt_mask)
                    dice_score = 2 * np.sum(intersection) / (np.sum(mask) + np.sum(gt_mask))
                    best_dice = max(best_dice, dice_score)
                
                ious.append(best_iou)
                dices.append(best_dice)

    print("Avg. IoU:", np.nanmean(ious))
    print("Avg. DICE: ", np.nanmean(dices))

if __name__ == "__main__":
    main()