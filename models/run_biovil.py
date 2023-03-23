from pathlib import Path
import torch

from .BioViL.text import get_cxr_bert_inference
from .BioViL.image import get_biovil_resnet_inference
from .BioViL.vlp import ImageTextInferenceEngine
from .BioViL.common.visualization import plot_phrase_grounding_similarity_map

text_inference = get_cxr_bert_inference()
image_inference = get_biovil_resnet_inference()

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)

def plot_phrase_grounding(image_path: str, text_prompt: str) -> None:
    similarity_map = image_text_inference.get_similarity_map_from_raw_data(
        image_path=Path(image_path),
        query_text=text_prompt,
        interpolation="bilinear",
    )
    plot_phrase_grounding_similarity_map(
        image_path=Path(image_path),
        similarity_map=similarity_map,
        fname="../../biovil_plot.png",
    )