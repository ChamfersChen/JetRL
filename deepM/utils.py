# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM visualization
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import requests
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor

from .torchcam import methods
from .torchcam.utils import overlay_mask

from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, resnet18_ml_backbone,\
                        resnet50_ml_backbone,resnet50_sg_backbone


def get_cam(img_path, 
            arch = 'resnet18',
            method = "GradCAM", 
            alpha = 0.5,
            rows = 1, 
            class_idx = 232, 
            savefig = "/Users/mac/Desktop/Projects/Django/jetimg/media/grad_cam.png",
            noblock = False,
            device = "cuda:0" if torch.cuda.is_available() else "cpu"):


    device = torch.device(device)

    # Pretrained imagenet model
    # ######加载自定义模型
    model = models.__dict__[arch](pretrained=True).to(device=device)
    # ##################

    # Image
    # if args.img.startswith("http"):
    #     img_path = BytesIO(requests.get(args.img).content)
    # else:
    #     img_path = args.img
    pil_img = Image.open(img_path, mode="r").convert("RGB")

    # Preprocess image
    img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(
        device=device
    )

    if isinstance(method, str):
        cam_methods = method.split(",")
    else:
        raise NotImplementedError

    # Hook the corresponding layer in the model
    cam_extractors = [methods.__dict__[name](model, enable_hooks=False) for name in cam_methods]

    # Homogenize number of elements in each row
    num_cols = math.ceil((len(cam_extractors) + 1) / rows)

    _, axes = plt.subplots(rows, num_cols, figsize=(6, 4))
    # Display input
    ax = axes[0][0] if rows > 1 else axes[0] if num_cols > 1 else axes
    ax.imshow(pil_img)
    ax.set_title("Input", size=8)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
        extractor._hooks_enabled = True
        model.zero_grad()
        scores = model(img_tensor.unsqueeze(0))

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item() if class_idx is None else class_idx

        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores)[0].squeeze(0).cpu()

        # Clean data
        extractor.remove_hooks()
        extractor._hooks_enabled = False
        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode="F")
        # Plot the result
        result = overlay_mask(pil_img, heatmap, alpha=alpha)

        ax = axes[idx // num_cols][idx % num_cols] if rows > 1 else axes[idx] if num_cols > 1 else axes

        ax.imshow(result)
        ax.set_title(extractor.__class__.__name__, size=8)

    # Clear axes
    if num_cols > 1:
        for _axes in axes:
            if rows > 1:
                for ax in _axes:
                    ax.axis("off")
            else:
                _axes.axis("off")

    else:
        axes.axis("off")

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    # plt.show(block=not noblock)

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_level_feature_map(view1, view2, img_prefix = "low",t = 2, gap = 4, num_p=3, media_path = '/Users/mac/Desktop/Projects/Django/jetimg/media/'):
    # 保存不同层级特征
    _, b, w, h = view1.shape

    n_v1 = view1[0].detach().numpy()
    n_v2 = view2[0].detach().numpy()

    cat_v1 = np.ones(((w+gap)*num_p,(h+gap)*num_p))
    cat_v2 = np.ones(((w+gap)*num_p,(h+gap)*num_p))

    idx = 0
    for j in range(num_p):
        for i in range(num_p):

            pix1 = np.ones((w+gap,h+gap))
            pix1[gap:w+gap, gap:h+gap] = n_v1[idx]*t
            
            cat_v1[i*w:(i+1)*w+gap, j*h:(j+1)*h+gap] = pix1
            
            pix2 = np.ones((w+gap,h+gap))
            pix2[gap:w+gap, gap:h+gap] = n_v2[idx]*t
            
            cat_v2[i*w:(i+1)*w+gap, j*h:(j+1)*h+gap] = pix2
            idx+=1

    cv2.imwrite(os.path.join(media_path, "{}_v1.png".format(img_prefix)), cat_v1*255)
    cv2.imwrite(os.path.join(media_path, "{}_v2.png".format(img_prefix)), cat_v2*255)