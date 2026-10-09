import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataset import get_dataloaders
from model import AirQualityLSTM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, scaler = get_dataloaders(batch_size=64)

    model = AirQualityLSTM(input_dim=8, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1)
    model_path = os.path.join("checkpoints", "best_lstm.pt")

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    actuals = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # squeeze the outputs to match targets dimension
            outputs = outputs.squeeze(-1)
            targets = targets.squeeze(-1) if targets.dim() > 1 else targets

            actuals.append(targets.cpu().numpy())
            predictions.append(outputs.cpu().numpy())

    actuals = np.concatenate(actuals)
    predictions = np.concatenate(predictions)

    # Note: actuals and predictions are scaled. To get real MAE/RMSE we need to inverse transform
    # The scaler was fit on all 8 features. We only care about pm2.5 which is index 0.
    # To inverse transform, we need an array of shape (N, 8).

    def inverse_transform_target(data, scaler):
        dummy = np.zeros((len(data), 8))
        dummy[:, 0] = data
        return scaler.inverse_transform(dummy)[:, 0]

    actuals_inv = inverse_transform_target(actuals, scaler)
    predictions_inv = inverse_transform_target(predictions, scaler)

    mae = mean_absolute_error(actuals_inv, predictions_inv)
    rmse = np.sqrt(mean_squared_error(actuals_inv, predictions_inv))

    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Ensure artifacts directory exists
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_inv[:200], label="Actual PM2.5", color='blue') # plot first 200 for readability
    plt.plot(predictions_inv[:200], label="Predicted PM2.5", color='red')
    plt.title("Actual vs Predicted PM2.5 (Test Set - First 200 Samples)")
    plt.xlabel("Time Step")
    plt.ylabel("PM2.5 Concentration")
    plt.legend()
    plt.savefig(os.path.join(artifacts_dir, "actual_vs_predicted.png"))
    plt.close()

    # Plot 2: Loss Curves
    losses_path = os.path.join("checkpoints", "losses.pt")
    if os.path.exists(losses_path):
        losses = torch.load(losses_path)
        train_losses = losses.get('train_losses', [])
        val_losses = losses.get('val_losses', [])

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training and Validation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.savefig(os.path.join(artifacts_dir, "loss_curves.png"))
        plt.close()
    else:
        print(f"Losses file not found at {losses_path}. Skipping loss curve generation.")

if __name__ == "__main__":
    main()
