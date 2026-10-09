import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import AirQualityLSTM

def main():
    parser = argparse.ArgumentParser(description="Train AirQualityLSTM")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")

    args = parser.parse_args()

    if args.smoke_test:
        print("Running in smoke-test mode...")
        args.epochs = 2
        args.batch_size = 64

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, _ = get_dataloaders(batch_size=args.batch_size)

    model = AirQualityLSTM(input_dim=8, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoints_dir, "best_lstm.pt")

    # Check if we should use limited batches for smoke-test just in case dataset is huge?
    # The requirement says "caps at 2 epochs, batch size 64 for instant verification".
    # I'll train standard way but just for 2 epochs.

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs shape is (batch, 1), targets shape is (batch, 1) or (batch,)
            loss = criterion(outputs.squeeze(-1), targets.squeeze(-1) if targets.dim() > 1 else targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            if args.smoke_test and batch_idx >= 5: # limit batches for instant verification
                break

        if args.smoke_test:
            # If smoke test, divide by the actual number of samples processed
            train_loss = train_loss / (6 * args.batch_size) # processed indices 0 to 5, total 6 batches
        else:
            train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1), targets.squeeze(-1) if targets.dim() > 1 else targets)
                val_loss += loss.item() * inputs.size(0)

                if args.smoke_test and batch_idx >= 5:
                    break

        if args.smoke_test:
            val_loss = val_loss / (6 * args.batch_size)
        else:
            val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val Loss: {val_loss:.4f}")

    torch.save({'train_losses': train_losses, 'val_losses': val_losses}, os.path.join(checkpoints_dir, "losses.pt"))

if __name__ == "__main__":
    main()
