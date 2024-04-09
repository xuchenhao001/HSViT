from typing import List

import torch
from torch import nn

from utils.visualizer import show_feature_maps


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class Conv2dBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool_stride: int = 1):
        super().__init__()
        max_pooler = nn.MaxPool2d(kernel_size=kernel_size, stride=pool_stride, padding=kernel_size // 2) \
            if pool_stride > 1 else nn.Identity()
        self.layer = nn.Sequential(
            Conv2dLayer(in_channels, out_channels, kernel_size=kernel_size),
            Conv2dLayer(out_channels, out_channels),
            max_pooler
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.layer(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
            Attention Block
            Ref: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html

            We use the Pre-Layer Normalization version of the Transformer blocks,
            proposed by Ruibin Xiong et al. "On layer normalization in the transformer architecture." PMLR, 2020.

        Args:
            embed_dim: int, dimension of embeddings
            num_heads: int, number of attention heads
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, input_q, input_k, input_v):
        """
        Args:
            input_q: torch.Tensor, (batch_size, num_tokens_q, embed_dim)
            input_k: torch.Tensor, (batch_size, num_tokens_kv, embed_dim)
            input_v: torch.Tensor, (batch_size, num_tokens_kv, embed_dim)

        Returns:
            torch.Tensor: (batch_size, num_tokens_q, embed_dim)
        """
        norm_q = self.layer_norm(input_q)
        norm_k = self.layer_norm(input_k)
        norm_v = self.layer_norm(input_v)
        hidden_states = input_q + self.self_attn(norm_q, norm_k, norm_v)[0]
        return hidden_states


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn_block = AttentionBlock(embed_dim, num_heads, dropout)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: torch.Tensor, (batch_size, num_tokens, embed_dim)

        Returns:
            torch.Tensor, (batch_size, num_tokens, embed_dim)
        """
        hidden_states = self.attn_block(hidden_states, hidden_states, hidden_states)
        return hidden_states


class HSViTModel(nn.Module):
    def __init__(
            self,
            num_channels: int,
            image_size: int,
            conv_kernel_nums: List[int],
            conv_kernel_sizes: List[int],
            pool_strides: List[int],
            attn_num: int,
            attn_depth: int,
            attn_embed_dim: int,
            num_heads: int,
            dropout: float,
    ):
        super().__init__()
        self.attn_num = attn_num
        self.attn_embed_dim = attn_embed_dim

        if conv_kernel_sizes[-1] == 0:
            self.enable_conv_layer = False
            self.conv_kernel_num = conv_kernel_nums[-1]
        else:
            self.enable_conv_layer = True
            # multiple layers of Conv2D
            kernel_num_list = [num_channels] + conv_kernel_nums
            self.conv2d_layers = nn.Sequential(*[
                Conv2dBasicLayer(in_kernel_num, out_kernel_num, conv_kernel_size, conv_stride)
                for in_kernel_num, out_kernel_num, conv_kernel_size, conv_stride in
                zip(kernel_num_list[:-1], kernel_num_list[1:], conv_kernel_sizes, pool_strides)
            ])

            feature_map_size = (image_size // torch.prod(torch.tensor(pool_strides))) ** 2
            self.embed_projection = nn.Sequential(
                nn.Linear(feature_map_size, attn_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        if attn_num > 0:
            assert conv_kernel_nums[-1] % attn_num == 0, "conv_kernel_nums should be divisible by attn_num"
            self.cls_token_embeddings = nn.Parameter(torch.randn(attn_num, 1, 1, attn_embed_dim))
            self.position_embeddings = nn.Parameter(
                torch.randn(attn_num, 1, conv_kernel_nums[-1] // attn_num + 1, attn_embed_dim)
            )

            self.attn_encoders = nn.ModuleList(
                [nn.Sequential(*[
                    SelfAttentionBlock(attn_embed_dim, num_heads, dropout) for _ in range(attn_depth)
                ]) for _ in range(attn_num)]
            )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: torch.Tensor, (batch_size, num_channels, height, width)

        Returns:
            torch.Tensor: (batch_size, attn_num, attn_embed_dim)
        """
        batch_size, num_channels, height, width = pixel_values.shape

        if self.enable_conv_layer:
            # apply multiple Conv2d layers to the image
            # conv2d_embeddings: (batch_size, conv_kernel_num, feature_map_h, feature_map_w)
            conv2d_embeddings = self.conv2d_layers(pixel_values)
            # visualize conv2d embeddings
            # show_feature_maps(pixel_values, conv2d_embeddings, self.attn_num)
            # exit(0)
            # embed_dim = feature_map_h * feature_map_w
            # conv2d_embeddings: (batch_size, conv_kernel_num, feature_map_h * feature_map_w)
            conv2d_embeddings = conv2d_embeddings.flatten(start_dim=2)
            # after projection, conv2d_embeddings: (batch_size, conv_kernel_num, attn_embed_dim)
            conv2d_embeddings = self.embed_projection(conv2d_embeddings)
        else:
            assert height * width == self.attn_embed_dim, "height x width must equal to attn_embed_dim"
            # conv2d_embeddings: (batch_size, num_channels, attn_embed_dim)
            conv2d_embeddings = pixel_values.flatten(start_dim=2)
            # after repeat, conv2d_embeddings: (batch_size, conv_kernel_num, attn_embed_dim)
            conv2d_embeddings = conv2d_embeddings.repeat(1, self.conv_kernel_num // num_channels + 1, 1)
            conv2d_embeddings = conv2d_embeddings[:, :self.conv_kernel_num, :]

        if self.attn_num > 0:  # if enable self-attention
            # split conv2d_embeddings to different Transformer encoders
            split_conv2d_embeddings = torch.tensor_split(conv2d_embeddings, self.attn_num, dim=1)
            # split_conv2d_embeddings: (attn_num, batch_size, conv_kernel_num // attn_num, attn_embed_dim)
            split_conv2d_embeddings = torch.stack(split_conv2d_embeddings, dim=0)

            # add CLS token to conv2d_embeddings
            cls_token_embeddings = self.cls_token_embeddings.repeat(1, batch_size, 1, 1)
            split_conv2d_embeddings = torch.cat([cls_token_embeddings, split_conv2d_embeddings], dim=2)
            # add positional information to conv2d_embeddings
            split_conv2d_embeddings = split_conv2d_embeddings + self.position_embeddings

            # go through self-attention encoders
            # split_conv2d_embeddings: (attn_num, batch_size, 1 + conv_kernel_num // attn_num, attn_embed_dim)
            hidden_states = []
            for conv2d_embeddings, attn_encoder in zip(split_conv2d_embeddings, self.attn_encoders):
                hidden_state = attn_encoder(conv2d_embeddings)
                hidden_states.append(hidden_state)
            # hidden_states: (batch_size, attn_num, 1 + conv_kernel_num // attn_num, attn_embed_dim)
            hidden_states = torch.stack(hidden_states, dim=1)

            # aggregate hidden_states from the first CLS tokens
            hidden_states = hidden_states[:, :, 0, :]  # (batch_size, attn_num, attn_embed_dim)
        else:
            hidden_states = conv2d_embeddings  # (batch_size, conv_kernel_num, attn_embed_dim)

        return hidden_states


class ClassificationHead(nn.Module):
    def __init__(self, attn_embed_dim: int, num_classes: int):
        """
        Perform classification prediction

        Args:
            attn_embed_dim: int
            num_classes: int
        """
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.mlp_head = nn.Sequential(nn.LayerNorm(attn_embed_dim), nn.Linear(attn_embed_dim, num_classes))

    def forward(self, x):
        """

        Args:
            x: torch.Tensor, (batch_size, num_tokens, attn_embed_dim)

        Returns:
            torch.Tensor, (batch_size, num_classes)
        """
        # (batch_size, num_tokens, attn_embed_dim) -> (batch_size, attn_embed_dim)
        hidden_states = self.pool(x.permute(0, 2, 1)).squeeze(2)
        logits = self.mlp_head(hidden_states)
        return logits


class HSViTForImageClassification(nn.Module):
    def __init__(
            self,
            num_channels: int,
            image_size: int,
            conv_kernel_nums: List[int],
            conv_kernel_sizes: List[int],
            pool_strides: List[int],
            attn_num: int,
            attn_depth: int,
            attn_embed_dim: int,
            num_heads: int,
            num_classes: int,
            dropout: float,
    ):
        super().__init__()
        self.model = HSViTModel(num_channels, image_size, conv_kernel_nums, conv_kernel_sizes, pool_strides,
                                attn_num, attn_depth, attn_embed_dim, num_heads, dropout)
        self.classifier = ClassificationHead(attn_embed_dim, num_classes)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: torch.Tensor, (batch_size, num_channels, height, width)

        Returns:
            torch.Tensor, (batch_size, num_classes)
        """
        hidden_states = self.model(pixel_values)
        logits = self.classifier(hidden_states)
        return logits

