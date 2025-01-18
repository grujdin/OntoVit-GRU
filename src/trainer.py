# src/trainer.py
import torch
import torch.nn as nn

def train_ontovit_gru(model, dataloader, optimizer, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()  # or e.g. BCELoss if binary
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            frames = batch['frames'].to(device)  # [B,C,T,H,W]
            label_mask = batch['label_mask'].to(device) # shape => [B, H, W], or [B] ...
            # depends on your use-case

            outputs = model(batch)
            # if it's classification => shape [B,2]
            # if it's segmentation => shape [B,2,H',W'], we'd need a different criterion
            loss = criterion(outputs, label_mask.long()) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
