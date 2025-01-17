def train_ontovit_gru(model, dataset, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        for frames, label_mask in dataset:
            # frames: [pre, flood, post], each => shape [B, c, h, w]
            outputs = model(frames)  # shape [B, #classes]
            loss = loss_fn(outputs, label_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
