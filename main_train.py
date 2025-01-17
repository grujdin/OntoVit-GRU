import torch
from torch.utils.data import DataLoader
from src.dataset import FloodEventDataset
from src.model import OntoViTGRU
from src.trainer import train_ontovit_gru

def main():
    # Load concept embeddings
    # Initialize dataset
    dataset = FloodEventDataset(...)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # Initialize model, optimizer, loss
    model = OntoViTGRU(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Train
    train_ontovit_gru(model, loader, optimizer, loss_fn)

if __name__ == "__main__":
    main()
