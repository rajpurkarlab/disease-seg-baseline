import json
from matplotlib import pyplot as plt
from pathlib import Path
from uuid import uuid4
import pycocotools.mask as mask_util
import argparse
import sys

from utils import compute_segmentation_metrics, PATH_TO_ID
from models.run_biovil import plot_phrase_grounding as ppgb
from models.BioViL.image.data.io import load_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of model (BioViL, )')
    parser.add_argument('test_set', type=str, help='name of test set (CheXlocalize, )')
    parser.add_argument('visualize', type=str, help='yes or no')
    parser.add_argument('method', type=str, help='how to generate heatmap (naive, grad_cam, gradcam_plus, cocoa)')
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

def main():
    args = parse_args()
    print(f"Running {sys.argv[0]} with args {args}")

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
            annots = json_obj[obj][query]

            if annots['counts'] != 'ifdl3':
                gt_mask = mask_util.decode(annots)
                if gt_mask.max() == 0:
                    continue
                text_prompt = PROMPTS[query]
                if args.method == "naive":
                    heatmap = ppgb(filename, text_prompt)
                else:
                    heatmap = ppgb(filename, text_prompt, method=args.method, input_size=gt_mask.shape, pathology=query)
                
                best_iou, best_dice, best_thresh = compute_segmentation_metrics(heatmap, gt_mask)
                
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
    
    f = open("run_seg.txt", "a")

    for pathology in PROMPTS:
        print("Pathology:", pathology, "mIoU:", ious_by_pathology[PATH_TO_ID[pathology]]/numbers_by_pathology[PATH_TO_ID[pathology]], "Avg. DICE:", dices_by_pathology[PATH_TO_ID[pathology]]/numbers_by_pathology[PATH_TO_ID[pathology]])
        f.write("Pathology: " + pathology + " mIoU: " + str(ious_by_pathology[PATH_TO_ID[pathology]]/numbers_by_pathology[PATH_TO_ID[pathology]]) + " Avg. DICE: " + str(dices_by_pathology[PATH_TO_ID[pathology]]/numbers_by_pathology[PATH_TO_ID[pathology]]) + "\n")

    f.write("\n")
    print("mIoU:", sum(ious_by_pathology)/sum(numbers_by_pathology))
    f.write("mIoU: " + str(sum(ious_by_pathology)/sum(numbers_by_pathology)) + "\n")
    print("Avg. DICE: ", sum(dices_by_pathology)/sum(numbers_by_pathology))
    f.write("Avg. DICE: " + str(sum(dices_by_pathology)/sum(numbers_by_pathology)) + "\n")

    f.close()

if __name__ == "__main__":
    main()