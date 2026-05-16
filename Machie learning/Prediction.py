import os
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



class FourByFourCNN(nn.Module):
    """Simple 1D CNN regression model.

    The first convolution layer uses 4 output channels and a kernel size of 4,
    corresponding to a 4-by-4 CNN-style first layer for one-dimensional signals.
    """

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=4, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x



class VOADataset(Dataset):
    """Dataset for loading one-dimensional signal samples and Acetone labels."""

    def __init__(self, dataset_dir, csv_path):
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, encoding="utf-8")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.loc[idx, "filepath"]
        x = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

        # Convert each sample to a single-channel sequence with shape (1, L).
        if x.ndim == 1:
            seq = x
        elif x.ndim == 2 and x.shape[1] >= 1:
            seq = x[:, 0]
        else:
            raise ValueError(f"Unsupported data shape from {file_path}: {x.shape}")

        seq = torch.from_numpy(seq).float().unsqueeze(0)

        # Apply sample-wise z-score normalization to improve training stability.
        mean = seq.mean(dim=1, keepdim=True)
        std = seq.std(dim=1, keepdim=True).clamp_min(1e-6)
        seq = (seq - mean) / std

        label = torch.tensor([self.df.loc[idx, "Acetone"]], dtype=torch.float32)
        return seq, label


def create_data_loaders(dataset, label_column="Acetone", test_size=0.2, batch_size=32, random_state=42):
    """Create stratified train/test data loaders."""
    y = dataset.df[label_column].values

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    test_idx = np.array(sorted(test_idx))

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_idx, test_idx


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=130, save_path="best_cnn_model.pt"):
    """Train the model and save the best checkpoint based on training loss."""
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        last_loss = None

        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

            if last_loss < best_loss:
                best_loss = last_loss
                torch.save(model.state_dict(), save_path)

        if last_loss is not None and last_loss < 1e-3:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {last_loss:.5f}")
            print("The loss threshold has been reached.")
            break

        if (epoch + 1) % 10 == 0 and last_loss is not None:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {last_loss:.5f}")

    return best_loss


def evaluate_model(model, test_loader, dataset, test_idx, device, label_column="Acetone"):
    """Evaluate the model on the test set and return predictions and metrics."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.float().to(device)
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy().reshape(-1))

    y_pred = np.concatenate(predictions, axis=0)
    y_true = dataset.df.loc[test_idx, label_column].to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    tolerances = [0.01, 0.05, 0.10]
    accuracy_within = {
        tol: float((np.abs(y_pred - y_true) <= tol).mean() * 100.0)
        for tol in tolerances
    }

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        **{f"Accuracy@±{tol}": acc for tol, acc in accuracy_within.items()},
    }

    return y_true, y_pred, metrics


def save_test_results(dataset, test_idx, y_pred, metrics, output_dir="results"):
    """Save test predictions and summary metrics."""
    os.makedirs(output_dir, exist_ok=True)

    test_meta = dataset.df.loc[test_idx, ["filename", "filepath", "Acetone"]].reset_index(drop=True)
    prediction_df = test_meta.copy()
    prediction_df["y_pred"] = y_pred
    prediction_df["abs_error"] = (prediction_df["y_pred"] - prediction_df["Acetone"]).abs()

    prediction_csv = os.path.join(output_dir, "test_predictions.csv")
    prediction_df.to_csv(prediction_csv, index=False)

    metrics_csv = os.path.join(output_dir, "test_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])

        for metric_name, metric_value in metrics.items():
            if metric_name.startswith("Accuracy"):
                writer.writerow([metric_name, f"{metric_value:.2f}%"])
            else:
                writer.writerow([metric_name, f"{metric_value:.6f}"])

    print(f"Test predictions saved to: {prediction_csv}")
    print(f"Test metrics saved to: {metrics_csv}")


def main():
    dataset_dir = "./processed_dataset"
    csv_path = "./EAI_label.csv"
    batch_size = 32
    learning_rate = 1.5e-4
    num_epochs = 130
    model_save_path = "best_cnn_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = VOADataset(dataset_dir, csv_path)
    print(f"Number of samples: {len(dataset)}")

    train_loader, test_loader, _, test_idx = create_data_loaders(
        dataset,
        label_column="Acetone",
        test_size=0.2,
        batch_size=batch_size,
        random_state=42,
    )

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    model = FourByFourCNN().float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_path=model_save_path,
    )

    y_true, y_pred, metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        dataset=dataset,
        test_idx=test_idx,
        device=device,
        label_column="Acetone",
    )

    print(
        f"[TEST] MAE={metrics['MAE']:.6f}  "
        f"MSE={metrics['MSE']:.6f}  "
        f"RMSE={metrics['RMSE']:.6f}  "
        f"R2={metrics['R2']:.6f}"
    )

    for metric_name, metric_value in metrics.items():
        if metric_name.startswith("Accuracy"):
            print(f"[TEST] {metric_name}: {metric_value:.2f}%")

    save_test_results(
        dataset=dataset,
        test_idx=test_idx,
        y_pred=y_pred,
        metrics=metrics,
        output_dir="results",
    )


if __name__ == "__main__":
    main()
