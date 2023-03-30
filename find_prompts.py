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
    parser.add_argument('validation_set', type=str, help='name of test set (CheXlocalize, )')
    parser.add_argument('corpus_set', type=str, help='name of corpus set (MIMIC-CXR, MS-CXR, )')
    return parser.parse_args()

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

    if args.model != "BioViL":
        raise NotImplementedError("Only BioViL is implemented for now")
    if args.validation_set != "CheXlocalize":
        raise NotImplementedError("Only CheXlocalize is implemented for now")
    
    if args.corpus_set == "MIMIC-CXR":
        pass
    elif args.corpus_set == "MS-CXR":
        pass
    else:
        raise NotImplementedError("Only MIMIC-CXR and MS-CXR are implemented for now")

    
if __name__ == "__main__":
    main()