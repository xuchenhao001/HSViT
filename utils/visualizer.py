import colorsys
import random
from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2


def generate_rdm_colors(num_colors: int):
    rdm_colors = []
    for _ in range(num_colors):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        rdm_rgb = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        rdm_colors.append(rdm_rgb)
    return np.array(rdm_colors).astype('uint8')


__DEFAULT_COLORMAP_1K = generate_rdm_colors(1000)


def img_denormalize(img: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tv_transform = v2.Compose([
        v2.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std],
            inplace=False,
        ),
        v2.ToPILImage(mode="RGB"),
    ])
    pil_image = tv_transform(img)
    return pil_image


def show_images(images: List[torch.Tensor]):
    """
    show each image in the list for debugging, need GUI

    :param images: (list of torch.Tensor, (C, H, W), float) – Normalized RGB images
    """
    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    plt.rcParams['figure.constrained_layout.use'] = True

    # plot images one-by-one
    for i, image in zip(range(len(images)), images):
        plt.subplot(1, len(images), i+1)
        # denormalize images to PIL images
        image = img_denormalize(image)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


def show_feature_maps(pixel_values, conv2d_embeddings: torch.Tensor, attn_num: int):
    """
    Args:
        pixel_values: torch.Tensor, (batch_size, num_channel, height, width)
        conv2d_embeddings: torch.Tensor, (batch_size, conv_kernel_num, feature_map_h, feature_map_w)
        attn_num:
    """

    split_conv2d_embeddings = torch.tensor_split(conv2d_embeddings, attn_num, dim=1)
    # split_conv2d_embeddings: (batch_size, attn_num, conv_kernel_num // attn_num, feature_map_h, feature_map_w)
    split_conv2d_embeddings = torch.stack(split_conv2d_embeddings, dim=1)

    fig = plt.figure(figsize=(30, 50))
    for i, batch_item in enumerate(split_conv2d_embeddings):
        print(f"Debug: visualize feature map {i}...")
        # batch_item: (attn_num, conv_kernel_num // attn_num, feature_map_h, feature_map_w)
        for j, attn_feature_map in enumerate(batch_item):
            # attn_feature_map: (conv_kernel_num // attn_num, feature_map_h, feature_map_w)
            gray_scale = torch.sum(attn_feature_map, 0)
            gray_scale = gray_scale / attn_feature_map.shape[0]  # (feature_map_h, feature_map_w)
            gray_scale = gray_scale.detach().numpy()

            fig.add_subplot(4, 4, j + 1)
            plt.imshow(gray_scale)
            plt.axis('off')
        plt.savefig(f"./outputs/visualized/feature-maps-{i}.jpg", bbox_inches='tight')
        plt.clf()
        # plot original image for reference
        plt.subplot()
        image = img_denormalize(pixel_values[i])
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f"./outputs/visualized/test-image-{i}.jpg", bbox_inches='tight')
        plt.clf()
