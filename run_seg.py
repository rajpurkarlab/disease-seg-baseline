import json
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from uuid import uuid4

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

    json_obj = json.load(open("datasets/CheXlocalize/gt_segmentations_test.json"))

    for obj in json_obj:
        filename = "datasets/CheXlocalize/CheXpert/test/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        for query in json_obj[obj]:
            text_prompt = PROMPTS[query]
            heatmap = ppgb(filename, text_prompt)
            THRESHOLD = np.nanmin(heatmap)+np.nanmax(heatmap)
            mask = (heatmap > THRESHOLD).astype(int)
           
            if PLOT_IMAGES:
                fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                image = load_image(Path(filename)).convert("RGB")
                axes[0].imshow(image)
                axes[0].axis('off')
                axes[0].set_title("Input image")

                axes[1].imshow(mask)
                axes[1].axis('off')
                axes[1].set_title(f"BioViL mask: {text_prompt}")
                plt.savefig(f"biovil_plot_{uuid4()}.png")
            
            

            break
        break

if __name__ == "__main__":
    main()