from pathlib import Path
import torch

from .BioViL.text import get_cxr_bert_inference
from .BioViL.image import get_biovil_resnet_inference
from .BioViL.vlp import ImageTextInferenceEngine
from .BioViL.common.visualization import plot_phrase_grounding_similarity_map

from .gradcam import get_gradcam_map
from .cocoa import get_cocoa_map

text_inference = get_cxr_bert_inference()
image_inference = get_biovil_resnet_inference()

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)

def plot_phrase_grounding(image_path, text_prompt, method="naive", input_size=None, pathology=None, seg_targets=None) -> None:
    if method == "naive":
        similarity_map = image_text_inference.get_similarity_map_from_raw_data(
            image_path=Path(image_path),
            query_text=text_prompt,
            interpolation="bilinear",
        )
    elif method == "grad_cam":
        similarity_map, img = get_gradcam_map(image_path, text_prompt, input_size, False, seg_targets)
    elif method == "gradcam_plus":
        similarity_map, img = get_gradcam_map(image_path, text_prompt, input_size, True, seg_targets)
    else:
        d = {
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
        foil_captions = list(set(list(d.keys())) - set([d[pathology]]))
        similarity_map = get_cocoa_map(image_path, text_prompt, foil_captions, input_size)
    return similarity_map, img