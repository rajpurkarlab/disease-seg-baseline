#!pip install ftfy regex tqdm matplotlib opencv-python scipy scikit-image
#!pip install git+https://github.com/openai/CLIP.git

import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn
from torch.autograd import Variable
import cv2


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def getAttMap(img, attn_map, blur=True):
    #resize image to (480,480)
    # img = cv2.resize(img, (480,480))

    # print(type(img))
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        # output = model.get_patchwise_projected_embeddings(input, True, permute=False) #model(input)
        output = model(input).projected_patch_embeddings
        print(output.shape)
        output = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)), 1).squeeze(-1).squeeze(-1)
        # output = Variable(output.data, requires_grad=True)
        print("SHAPES=====")
        print(output.shape, target.shape)
        print("=====")

        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    print(gradcam.shape, input.shape)
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

# !pip install git+https://github.com/rvignav/hi-ml-multimodal.git

import torch

from models.BioViL.text.utils import get_cxr_bert_inference as get_bert_inference
# from models.BioViL.text.utils import BertEncoderType
from models.BioViL.image.utils import get_biovil_resnet_inference as get_image_inference
# from models.BioViL.image.utils import ImageModelType

def get_gradcam_map(image_path, image_caption):
    text_inference = get_bert_inference()
    image_inference = get_image_inference()

    clip_model = "RN50"
    saliency_layer = "layer4"

    blur = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_input = image_inference.transform(Image.open(image_path).convert('L')).unsqueeze(0).to(device)
    image_np = load_image(image_path, 480)
    # text_input = clip.tokenize([image_caption]).to(device)

    attn_map = gradCAM(
        image_inference.model.to(device), #.encoder.encoder,
        image_input.to(device),
        text_inference.get_embeddings_from_prompt(image_caption).float().to(device),
        getattr(image_inference.model.encoder.encoder, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    # viz_attn(image_np, attn_map, blur)

    return attn_map