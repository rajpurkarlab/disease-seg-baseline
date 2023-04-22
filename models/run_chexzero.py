import torch
import requests
import os
from CheXzero.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import sys

from .CheXzero.zero_shot import load_chexzero

def plot_phrase_grounding(image_path, text_prompt, plot=False):
    if os.path.exists('weights/'):
        pass
    else:
        os.system("wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip")
        os.system("unzip -d weights -j weights.zip")

    # load model
    model = CLIPDensePredT(version='ViT-B/32', reduce_dim=64)
    model.eval();

    # _, transform = load_chexzero("models/CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_16000_0.861.pt") #clip.load(version, device='cpu', jit=False)

    # print("TRANSFORM: ", transform)

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

    # load and normalize image
    input_image = Image.open(image_path).convert('RGB')

    # print(transforms.ToTensor()(input_image).shape)

    # TODO (vramesh, 2023-3-23): Replace with CheXzero's transform?
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[101.48761,101.48761,101.48761], std=[83.43944,83.43944,83.43944]),
    #     transforms.Resize(size=(352,352), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None)
    # ])
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])

    img = transform(input_image).unsqueeze(0)

    # print(img.shape)

    prompts = [text_prompt]

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

    if plot:
        # visualize prediction
        _, ax = plt.subplots(1, 1+len(prompts), figsize=(15, 4))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(input_image)
        [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
        [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))];

        plt.savefig("chexzero_plot.png")

    return torch.sigmoid(preds[0][0])

# print("Running chexzero")
# plot_phrase_grounding("../datasets/CheXlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg", "Support device", plot=True)