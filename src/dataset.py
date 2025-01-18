# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from src.ontology_embeddings import load_ontology_embeddings, get_ontology_vector_for_patch
import numpy as np

class FloodEventDataset(Dataset):
    def __init__(self, data_root, concept_embeddings, transform=None):
        super().__init__()
        self.data_root = data_root
        self.concept_embeddings = concept_embeddings
        self.transform = transform
        # Gather a list of flood events, each with pre/active/post images, metadata...
        self.events = self._scan_data()

    def _scan_data(self):
        # e.g., parse directory structure or a JSON listing
        # return a list of dicts: 
        # [ { 'pre':..., 'active':..., 'post':..., 'mask':..., 'lat':..., 'lon':..., 'year':..., 'day_of_year':...}, ... ]
        pass

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        info = self.events[idx]
        # Load pre/active/post 
        # shape => (C,H,W)
        pre_img = self._load_image(info['pre'])
        active_img = self._load_image(info['active'])
        post_img = self._load_image(info['post'])
        # stack => shape (C,3,H,W)
        frames = np.stack([pre_img, active_img, post_img], axis=1)
        # => shape => (C,T,H,W), but you want (B,C,T,H,W)
        frames_torch = torch.tensor(frames, dtype=torch.float32)
        # mask
        label_mask = self._load_image(info['mask']) if 'mask' in info else None
        # For each patch, or region -> ontology. Typically done inside the model or a special step
        # or we store 'is_burned', 'imperv_ratio' etc. in info 
        lat, lon = info.get('lat', 0.), info.get('lon', 0.)
        year, day_of_year = info.get('year', 2024), info.get('day_of_year', 100)

        return {
          'frames': frames_torch, # shape => (C,T,H,W)
          'label_mask': label_mask, 
          'lat': lat,
          'lon': lon,
          'year': year,
          'day_of_year': day_of_year
        }

    def _load_image(self, path):
        # TODO: load e.g. from .npy or .tif
        # shape => (C,H,W)
        pass
