import torch
import requests
import os
from .CheXzero.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

def plot_phrase_grounding(image_path, text_prompt):
    os.system("wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip")
    os.system("unzip -d weights -j weights.zip")

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval();

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

    # load and normalize image
    input_image = Image.open(image_path)

    # TODO (vramesh, 2023-3-23): Replace with CheXzero's transform?
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    img = transform(input_image).unsqueeze(0)

    prompts = [text_prompt]

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

    # visualize prediction
    _, ax = plt.subplots(1, 1+len(prompts), figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
    [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))];