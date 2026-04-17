"""Decoder for visualizing [CLS] token embeddings.

This lightweight transformer decoder is used for diagnostic visualization only.
It reconstructs images from [CLS] token embeddings to inspect what visual information
is retained in the representation.

Usage during training:
    1. Set decoder.enabled: True in config
    2. Add VisualizationCallback to callbacks
    3. Reconstructed images will be logged to WandB

Usage after training:
    Run scripts/visualize_decoder.py with checkpoint path to generate visualizations.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class CrossAttention(nn.Module):
    """Cross-attention layer for decoder."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context):
        """
        x: query tokens (B, N_q, D)
        context: key/value tokens (B, N_kv, D)
        """
        x = self.norm_q(x)
        context = self.norm_kv(context)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DecoderBlock(nn.Module):
    """Decoder block with cross-attention and residual MLP."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.cross_attn = CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, context):
        x = x + self.cross_attn(self.norm(x), context)
        x = x + self.mlp(x)
        return x


class Decoder(nn.Module):
    """Lightweight transformer decoder for visualization.

    Decodes [CLS] token embedding into image patches.
    """

    def __init__(
        self,
        cls_dim=192,
        hidden_dim=256,
        num_patches=196,
        patch_size=16,
        depth=4,
        heads=8,
        dim_head=32,
        mlp_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Project [CLS] embedding to hidden dimension
        self.cls_proj = nn.Linear(cls_dim, hidden_dim)

        # Learnable query tokens (one per patch)
        self.query_tokens = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        # Decoder blocks
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                DecoderBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Project to patch pixels
        self.to_pixels = nn.Linear(hidden_dim, patch_size * patch_size * 3)

    def forward(self, cls_emb):
        """
        cls_emb: (B, D) [CLS] token embedding
        Returns: (B, 3, H, W) reconstructed image
        """
        batch_size = cls_emb.size(0)

        # Project [CLS] to hidden dimension and use as context
        context = self.cls_proj(cls_emb)  # (B, hidden_dim)
        context = context.unsqueeze(1)  # (B, 1, hidden_dim)

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_patches, hidden_dim)

        # Apply decoder blocks
        x = queries
        for block in self.blocks:
            x = block(x, context)

        x = self.norm(x)

        # Project to pixels
        patches = self.to_pixels(x)  # (B, num_patches, patch_size^2 * 3)

        # Rearrange to image
        num_patches_side = int(patches.size(1) ** 0.5)
        image = rearrange(
            patches,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=num_patches_side,
            w=num_patches_side,
            p1=self.patch_size,
            p2=self.patch_size,
            c=3,
        )

        return image