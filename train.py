#!/usr/bin/env python3


def main(
    x_file: str = "x.npy",
    y_file: str = "y.npy",
    output_file: str = "output.pth",
    validation_split: float = 0.05,
    batch_size: int = 32,
    num_workers: int = 4,
    epochs: int = 100,
    resume_from: str = None,
):
    from modules.mlp import MLP
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    print("Loading dataset...")
    x = np.load(x_file)
    y = np.load(y_file)

    train_border = int(len(x) * (1 - validation_split))

    train_tensor_x = torch.Tensor(x[:train_border])
    train_tensor_y = torch.Tensor(y[:train_border])
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_tensor_x = torch.Tensor(x[train_border:])
    val_tensor_y = torch.Tensor(y[train_border:])
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    print("Setting up models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(768)

    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        model.load_state_dict(torch.load(resume_from))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    print("Training...")
    model.train()
    best_loss = float("inf")

    for epoch in tqdm(range(epochs)):
        total_loss_train = 0
        for batch_num, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x, batch_y = batch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
        print(f"Epoch {epoch + 1} | Loss {total_loss_train / len(train_loader)}")

        total_loss_val = 0
        for batch_num, batch in enumerate(val_loader):
            optimizer.zero_grad()

            batch_x, batch_y = batch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            total_loss_val += loss.item()

        avg_loss_val = total_loss_val / len(val_loader)
        print(f"Epoch {epoch + 1} | Val Loss {avg_loss_val}")

        if avg_loss_val < best_loss:
            print("Saving model...")
            torch.save(model.state_dict(), output_file)
            best_loss = avg_loss_val
            print("Saved model.")

    print("Done.")


if __name__ == "__main__":
    import typer

    typer.run(main)
