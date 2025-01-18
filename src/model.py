# src/model.py
import torch
import torch.nn as nn
from src.mae_utils import PatchEmbed3D, get_3d_sincos_pos_embed, TemporalEncoder, LocationEncoder
# or we might do huggingface from "transformers import ViTModel"

class OntoViTEncoder(nn.Module):
    def __init__(self, embed_dim=768, in_chans=3):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            input_size=(3, 224,224), # T=3 frames, 224x224
            patch_size=(1,16,16),
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        # Could add an actual Transformer block, or just do a simpler approach
        # etc...
        self.pos_embed = None # or define

    def forward(self, x, year=None, day_of_year=None, lat=None, lon=None):
        # x => shape [B, C, T, H, W]
        patches = self.patch_embed(x)
        # shape => [B, num_patches, embed_dim]
        # Possibly add sin/cos pos embeddings or temporal encoder
        return patches

class OntoViTGRU(nn.Module):
    def __init__(self, embed_dim=768, hidden_size=256, in_chans=3):
        super().__init__()
        self.encoder = OntoViTEncoder(embed_dim=embed_dim, in_chans=in_chans)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        # final head (e.g., 2 classes => flood/no flood)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, frames_batch):
        """
        frames_batch => dict with frames shape [B, C, T, H, W], plus optional lat/lon/year/doy
        """
        x = self.encoder(frames_batch['frames'],
                         year=frames_batch.get('year'),
                         day_of_year=frames_batch.get('day_of_year'),
                         lat=frames_batch.get('lat'),
                         lon=frames_batch.get('lon'))
        # x => shape [B, num_patches, embed_dim]
        # We treat num_patches as "sequence" dimension for GRU
        out, _ = self.gru(x) # shape => [B, num_patches, hidden_size]
        # final = last patch as "summary"
        final = out[:, -1, :]
        logits = self.head(final)
        return logits
