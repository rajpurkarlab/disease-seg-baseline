from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from .rise import RISE
from .contrastive_corpus_similarity import ContrastiveCorpusSimilarity

from models.BioViL.text.utils import get_cxr_bert_inference as get_bert_inference
from models.BioViL.image.utils import get_biovil_resnet_inference as get_image_inference

text_inference = get_bert_inference()
image_inference = get_image_inference()

def get_cocoa_map(image_path, image_caption, foil_captions, input_size):
    # Load explicand image
    explicand = image_inference.transform(Image.open(image_path).convert('L')).unsqueeze(0)
    
    # Set up baseline
    baseline = transforms.GaussianBlur(kernel_size=(5, 9), sigma=4)(explicand)

    # Turn corpus and foil captions into dataloaders of tokens
    corpus_tokens = [image_caption]
    # corpus_tokens = text_inference.get_embeddings_from_prompt(corpus_captions).float()
    # corpus_dataloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(corpus_tokens, torch.zeros(len(corpus_tokens))) # TensorDataset encoding requires labels => pass in dummy values
    # )

    foil_tokens = foil_captions
    # foil_dataloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(foil_tokens, torch.zeros(len(foil_tokens)))
    # )

    # Compute contrastive corpus similarity
    ccs = ContrastiveCorpusSimilarity(
        text_inference.get_embeddings_from_prompt,
        image_inference.model,
        corpus_tokens, 
        foil_tokens
    )

    # Generate COCOA attribution by passing contrastive corpus similarity to RISE
    attribution = RISE(ccs).rise(explicand, baseline)
    attribution = attribution[0].mean(0).numpy()

    print(attribution.shape)

    #resize to input size
    attribution = np.array(Image.fromarray(attribution).resize(input_size, Image.BILINEAR))

    return (attribution > (attribution.max() + attribution.min()) / 2).astype(int) # Need to threshold attribution since it takes on a smaller range of values than [0,1]