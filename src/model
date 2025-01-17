import torch
import torch.nn as nn
from transformers import ViTModel

class OntoViTGRU(nn.Module):
    def __init__(self, vit_model_name="ibm-nasa-geospatial/Prithvi-EO-2.0", hidden_size=256):
        super().__init__()
        # Load a pretrained ViT from huggingface or a local model path
        self.vit = ViTModel.from_pretrained(vit_model_name)
        # Freeze or partially unfreeze as needed
        # GRU for temporal dimension
        self.gru = nn.GRU(input_size=self.vit.config.hidden_size,
                           hidden_size=hidden_size,
                           batch_first=True)
        # A final segmentation/class head
        self.head = nn.Linear(hidden_size, 2)  # e.g., binary flood vs non-flood

    def forward(self, frames):
        """
        frames: a list of T frames,
                each shape [batch, channels, height, width] or tokens
        """
        all_outputs = []
        for frm in frames:
            # If raw imagery, patchify or convert to tokens
            # Example: Let the vit forward handle patching if using huggingface
            vit_out = self.vit(pixel_values=frm).last_hidden_state
            # Some shape: [batch, n_patches, hidden_size]
            all_outputs.append(vit_out)

        # stack them for GRU
        # shape => [batch, T, n_patches, hidden_size] or flatten n_patches dimension
        # Suppose we flatten n_patches dimension:
        # Or pass each patch through GRU if you prefer
        # For simplicity, let's do an average:
        # out_t = average across patch dimension => [batch, hidden_size]
        final_seq = []
        for t_out in all_outputs:
            out_t = t_out.mean(dim=1)
            final_seq.append(out_t)
        final_seq = torch.stack(final_seq, dim=1) # [batch, T, hidden_size]

        gru_out, _ = self.gru(final_seq)  # [batch, T, hidden_size]
        # take final time step
        final_out = gru_out[:, -1, :]
        logits = self.head(final_out)
        return logits
