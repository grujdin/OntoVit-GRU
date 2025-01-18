# src/mae_utils.py
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Copy the 1D sin/cos code from mae.py ...
    """
    # [Same code as mae.py]
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    3D sin/cos positional embeddings logic from mae.py
    """
    # [Same code from mae.py, or a simplified version]
    return pos_embed

class PatchEmbed3D(nn.Module):
    """
    A 3D patch embed, adapted from mae.py's PatchEmbed.
    Converts shape (B,C,T,H,W) -> (B,num_patches,embed_dim).
    """
    def __init__(self, input_size=(1,224,224), patch_size=(1,16,16),
                 in_chans=3, embed_dim=768, flatten=True):
        super().__init__()
        # [Initialize conv3d, etc. same logic as mae.py's PatchEmbed]
        # Possibly skip norm, or keep it
        # Example:
        self.flatten = flatten
        self.patch_size = patch_size
        # A simple 3D conv
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        # compute self.num_patches, self.grid_size, etc.

    def forward(self, x):
        # x shape => (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T//pt, H//ph, W//pw)
        if self.flatten:
            x = x.flatten(2).transpose(1,2) # (B, #patches, embed_dim)
        return x

class TemporalEncoder(nn.Module):
    """
    Copy from mae.py, adapted for your pipeline:
    """
    def __init__(self, embed_dim, trainable_scale=False):
        super().__init__()
        # [initialize, same logic]
        # self.scale = ...
        # self.year_embed_dim = ...
        # self.julian_day_embed_dim = ...
        pass

    def forward(self, temporal_coords, tokens_per_frame=None):
        """
        temporal_coords => shape (B, T, 2) => [year, day_of_year].
        Rescale using sin/cos, multiply by self.scale, etc.
        If tokens_per_frame is not None => expand embedding
        """
        # [similar to mae.py]
        pass

# etc. for LocationEncoder, or any other relevant classes from mae.py
