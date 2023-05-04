import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
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
    model,
    input,
    target,
    layer,
    input_size,
    gradcam_plus
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
        output = model(input).projected_patch_embeddings
        output = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)), 1).squeeze(-1).squeeze(-1)
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
        if gradcam_plus:
            gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input_size, #input.shape[2:]
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

from models.BioViL.text.utils import get_cxr_bert_inference as get_bert_inference
from models.BioViL.image.utils import get_biovil_resnet_inference as get_image_inference

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchxrayvision as xrv

model = xrv.baseline_models.chestx_det.PSPNet()

def get_gradcam_map(image_path, image_caption, input_size, gradcam_plus, seg_targets):
    text_inference = get_bert_inference()
    image_inference = get_image_inference()

    saliency_layer = "layer4"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    idxs = {
        "Left Lung": 4,
        "Right Lung": 5,
        "Heart": 8,
        "Facies Diaphragmatica": 10,
    }

    img = Image.open(image_path).convert('L')
    oimg = np.array(img)
    img = xrv.datasets.normalize(oimg, 255)
    img = torch.from_numpy(img)

    origimg = np.array(Image.open(image_path).convert('L'))

    if len(seg_targets) > 0:
        with torch.no_grad():
            pred = model(img)
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1

        preds = []

        for p in seg_targets:
            preds.append(np.array(Image.fromarray(pred[0, idxs[p]].numpy()).resize((input_size[1], input_size[0]), Image.BILINEAR)))
        
        # print(input_size, origimg.shape, preds[0].shape)

        for i in range(oimg.shape[0]):
            for j in range(oimg.shape[1]):
                not_in_either = True
                counter = 0
                for p in seg_targets:
                    predi = preds[counter]
                    counter += 1
                    
                    if predi[i, j] == 1:
                        not_in_either = False
                if not_in_either:
                    origimg[i, j] *= 0.7
        
    image_input = image_inference.transform(Image.fromarray(origimg)).unsqueeze(0).to(device)
    image_np = load_image(image_path, input_size[0])

    attn_map = gradCAM(
        image_inference.model.to(device), #.encoder.encoder,
        image_input.to(device),
        text_inference.get_embeddings_from_prompt(image_caption).float().to(device),
        getattr(image_inference.model.encoder.encoder, saliency_layer),
        input_size,
        gradcam_plus
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    map = getAttMap(image_np, attn_map, blur=True)
    if gradcam_plus:
        return map,origimg
    else:
        # invert map if most pixels are 0, else keep it same
        map = (map > (map.min() + map.max())/2).astype(int)
        if np.mean(map) < 0.5:
            return map, origimg
        else:
            return 1 - map, origimg