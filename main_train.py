# main_train.py
import torch
from torch.utils.data import DataLoader
from src.dataset import FloodEventDataset
from src.ontology_embeddings import load_ontology_embeddings
from src.model import OntoViTGRU
from src.trainer import train_ontovit_gru

def main():
    concept_embeddings = load_ontology_embeddings()
    # Create dataset
    dataset = FloodEventDataset(data_root='./data/example_flood_event',
                                concept_embeddings=concept_embeddings)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Create model
    model = OntoViTGRU(embed_dim=768, hidden_size=256, in_chans=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train_ontovit_gru(model, dataloader, optimizer, epochs=5)

if __name__ == "__main__":
    main()
