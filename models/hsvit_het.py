import torch
from torch import nn
from transformers import AutoModel


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


class HSViTHETModel(nn.Module):
    def __init__(
            self,
            sub_module_path: str,
            sub_module_feature_size: int,
            conv_kernel_num: int,
            enable_conv_layer: bool,
            attn_num: int,
            attn_depth: int,
            attn_embed_dim: int,
            num_heads: int,
            dropout: float,
    ):
        super().__init__()
        self.attn_num = attn_num
        self.attn_embed_dim = attn_embed_dim
        self.enable_conv_layer = enable_conv_layer

        if not enable_conv_layer:
            self.conv_kernel_num = conv_kernel_num
        else:
            self.conv2d_submodule = AutoModel.from_pretrained(sub_module_path)
            self.embed_projection = nn.Sequential(
                nn.Linear(sub_module_feature_size, attn_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        if attn_num > 0:
            assert conv_kernel_num % attn_num == 0, "conv_kernel_nums should be divisible by attn_num"
            self.cls_token_embeddings = nn.Parameter(torch.randn(attn_num, 1, 1, attn_embed_dim))
            self.position_embeddings = nn.Parameter(
                torch.randn(attn_num, 1, conv_kernel_num // attn_num + 1, attn_embed_dim)
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
            conv2d_output = self.conv2d_submodule(pixel_values=pixel_values, return_dict=True)
            # print(f"Debug conv2d_output.last_hidden_state: {conv2d_output.last_hidden_state.shape}")
            # exit(0)
            # conv2d_embedding: (batch_size, conv_kernel_num, feature_map_h, feature_map_w)
            conv2d_embeddings = conv2d_output.last_hidden_state
            # conv2d_embedding: (batch_size, conv_kernel_num, feature_map_h * feature_map_w)
            conv2d_embeddings = conv2d_embeddings.flatten(start_dim=2)
            # after projection, conv2d_embeddings: (batch_size, conv_kernel_num, attn_embed_dim)
            conv2d_embeddings = self.embed_projection(conv2d_embeddings)
            # print(f"Debug conv2d_embeddings: {conv2d_embeddings.shape}")

            # visualize conv2d embeddings
            # show_feature_maps(pixel_values, conv2d_embeddings, self.attn_num)
            # exit(0)
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


class HSViTHETForImageClassification(nn.Module):
    def __init__(
            self,
            sub_module_path: str,
            sub_module_feature_size: int,
            conv_kernel_num: int,
            enable_conv_layer: bool,
            attn_num: int,
            attn_depth: int,
            attn_embed_dim: int,
            num_heads: int,
            num_classes: int,
            dropout: float,
    ):
        super().__init__()
        self.model = HSViTHETModel(sub_module_path, sub_module_feature_size, conv_kernel_num, enable_conv_layer,
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
