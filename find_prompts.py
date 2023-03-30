import json
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from uuid import uuid4
import pycocotools.mask as mask_util
import argparse

from models.run_biovil import plot_phrase_grounding as ppgb
from models.BioViL.image.data.io import load_image
from utils import compute_segmentation_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of model (BioViL, )')
    parser.add_argument('validation_set', type=str, help='name of test set (CheXlocalize, )')
    parser.add_argument('corpus_set', type=str, help='name of corpus set (MIMIC-CXR, MS-CXR, )')
    return parser.parse_args()

def main():
    args = parse_args()

    PROMPTS_BY_PATH = {
        "Enlarged Cardiomediastinum": [],
        "Cardiomegaly": [],
        "Lung Lesion": [],
        "Airspace Opacity": [],
        "Edema": [],
        "Consolidation": [],
        "Atelectasis": [],
        "Pneumothorax": [],
        "Pleural Effusion": [],
        "Support Devices": []
    }

    if args.model != "BioViL":
        raise NotImplementedError("Only BioViL is implemented for now")
    if args.validation_set != "CheXlocalize":
        raise NotImplementedError("Only CheXlocalize is implemented for now")
    
    json_obj = json.load(open("datasets/CheXlocalize/gt_segmentations_val.json"))

    for obj in json_obj:
        filename = "datasets/CheXlocalize/CheXpert/val/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        for query in json_obj[obj]:
            annots = json_obj[obj][query]

            if annots['counts'] != 'ifdl3':
                gt_mask = mask_util.decode(annots)

                best_iou = 0
                best_prompt = ""

                if args.corpus_set == "MIMIC-CXR":
                    pass
                elif args.corpus_set == "MS-CXR":
                    # Loop through all MS-CXR (image,text) pairs, set text_prompt = text
                        # heatmap = ppgb(filename, text_prompt)
                        # iou, dice, _ = compute_segmentation_metrics(heatmap, gt_mask)
                        # if iou > best_iou:
                        #     best_iou = iou
                        #     best_prompt = text_prompt
                    pass
                else:
                    raise NotImplementedError("Only MIMIC-CXR and MS-CXR are implemented for now")
                
                PROMPTS_BY_PATH[query].append(best_prompt)

    
if __name__ == "__main__":
    main()