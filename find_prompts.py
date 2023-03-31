import json
import pycocotools.mask as mask_util
import argparse
import csv

from models.run_biovil import plot_phrase_grounding as ppgb
from models.BioViL.image.data.io import load_image
from utils import compute_segmentation_metrics, read_prompts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of model (BioViL, )')
    parser.add_argument('validation_set', type=str, help='name of test set (CheXlocalize, )')
    parser.add_argument('corpus_set', type=str, help='name of corpus set (MIMIC-CXR, MS-CXR, )')
    return parser.parse_args()

def main():
    args = parse_args()

    pathologies = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Support Devices"]

    best_ious = [0]*len(pathologies)
    best_prompts = ['']*len(pathologies)

    if args.model != "BioViL":
        raise NotImplementedError("Only BioViL is implemented for now")
    if args.validation_set != "CheXlocalize":
        raise NotImplementedError("Only CheXlocalize is implemented for now")
    
    json_obj = json.load(open("datasets/CheXlocalize/gt_segmentations_val.json"))

    tried_prompts = set()

    for pathology in pathologies:
        print(f"\n{pathology}\n")
        if args.corpus_set == "MIMIC-CXR":
            raise NotImplementedError("MIMIC-CXR not implemented yet")
        elif args.corpus_set == "MS-CXR":
            with open('datasets/MS-CXR/MS_CXR_Local_Alignment_v1.0.0.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if row[1] == pathology:
                        text_prompt = row[2]

                        if text_prompt in tried_prompts:
                            continue

                        tried_prompts.add(text_prompt)

                        tiou = 0
                        count = 0.0
                         
                        for obj in json_obj:
                            filename = "datasets/CheXlocalize/CheXpert/val/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"

                            annots = json_obj[obj][pathology]

                            if annots['counts'] != 'ifdl3':
                                heatmap = ppgb(filename, text_prompt)
                                gt_mask = mask_util.decode(annots)
                                iou, _, _ = compute_segmentation_metrics(heatmap, gt_mask)
                                tiou += iou
                                count += 1.0

                        if tiou/count > best_ious[pathologies.index(pathology)]:
                            best_ious[pathologies.index(pathology)] = tiou/count
                            best_prompts[pathologies.index(pathology)] = text_prompt            
        else:
            raise NotImplementedError("Only MIMIC-CXR and MS-CXR are implemented for now")

    f = open("results.txt", "a")
    f.write(str(best_ious))
    f.write("\n")
    f.write(str(best_prompts))
    f.close()
        
if __name__ == "__main__":
    main()