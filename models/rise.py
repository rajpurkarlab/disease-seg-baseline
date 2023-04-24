"""
Petsiuk et al., 2018, RISE: Randomized Input Sampling for Explanation of Black-Box Models (https://arxiv.org/abs/1806.07421)

This code was inspired by https://facebookresearch.github.io/TorchRay/_modules/torchray/attribution/rise.html. I essentially
reproduce the RISE code from TorchRay, but with a few changes (commented below as [CHANGE]) as per the implementation
decisions in the COCOA paper.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .contrastive_corpus_similarity import ContrastiveCorpusSimilarity

class RISE():
    """
    Code for the RISE feature attribution method.
    """

    def __init__(self, ccs: ContrastiveCorpusSimilarity): # [CHANGE] Pass in ContrastiveCorpusSimilarity instance instead of an actual encoder
        """
        Constructor for `RISE`.

        Parameters:
            ccs (ContrastiveCorpusSimilarity): ContrastiveCorpusSimilarity instance; target that RISE will explain
        """
        self.ccs = ccs

    def rise(
        self,
        inputs,
        baselines, # [CHANGE]
        num_masks = 10 # 20000
    ):
        """
        Performs RISE for pixel-wise feature attribution.

        Parameters:
            inputs (torch.Tensor): An input image for which a RISE saliency map is generated
            baselines (torch.Tensor): Baseline values with which to replace masked pixels in the input image
                As per the COCOA paper, this corresponds to the original image blurred with a Gaussian filter
            num_masks (int): number of RISE random masks to use. The COCOA paper uses 20,000 masks

        Returns:
            RISE saliency map (torch.Tensor), with the same shape as the input: `(batch_size, channel_size, image_height, image_width)`
        """

        batch_size, channel_size, image_height, image_width = inputs.shape
        saliency = torch.zeros(batch_size, channel_size, image_height, image_width) # Initialize "empty" saliency map

        for i in range(num_masks): # Generate 20,000 masks and use them to compute the RISE saliency map
            masks = self.generate_rise_masks(image_height, image_width, batch_size)
            print(inputs.shape, masks.shape, baselines.shape)
            try:
                importance_score = self.ccs.compute_contrastive_corpus_similarity(inputs * masks + baselines * (1 - masks)).detach() # [CHANGE] Replace masked input pixels with baseline pixels
                importance_score = importance_score.view(batch_size, 1, 1, 1) * masks # Reshape importance score to match the shape of the masks
                saliency += importance_score
            except:
                i -= 1
        
        saliency /= num_masks
        saliency /= 0.5 # Normalize by the probability (0.5) of masking a pixel in the smaller binary mask
        
        return saliency

    def generate_rise_masks(self, image_height, image_width, batch_size):
        """
        Generates RISE random masks and upsamples them (via bilinear interpolation) on the fly.

        Parameters:
            img_height (int): Original image height
            img_width (int): Original image width
            batch_size (int): Batch size, which equals number of masks to generate

        Returns
        -------
            Independently sampled masks (torch.Tensor)
        """

        grid_height = image_height // 7
        grid_width = image_width // 7
        mask = (torch.rand(batch_size, 1, 7, 7) < 0.5).float() # [CHANGE] Use 7x7 grid size
        mask = F.interpolate(mask, size=(8 * grid_height, 8 * grid_width), mode="bilinear", align_corners=False)
        
        # Save final RISE masks with random shift
        shift_x, shift_y = np.random.randint(grid_height), np.random.randint(grid_width)
        mask = mask[:, :, shift_x:(shift_x + image_height), shift_y:(shift_y + image_width)]

        return mask