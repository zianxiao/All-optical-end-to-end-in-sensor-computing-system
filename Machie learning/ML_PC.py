import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# File utilities
# ============================================================

def build_file_index(dataset_dir: Path) -> dict:
    """
    Recursively index CSV files under the dataset folder.

    This fallback is used when the label CSV contains only filenames.
    In this case, all filenames must be unique.
    """
    file_index = {}

    for csv_file in dataset_dir.rglob("*.csv"):
        filename = csv_file.name

        if filename in file_index:
            raise ValueError(
                f"Duplicate filename found: {filename}. "
                "Please provide a 'relative_path' column in the label CSV."
            )

        file_index[filename] = csv_file

    return file_index


def load_numeric_csv(file_path: Path, expected_length: int = 2000) -> np.ndarray:
    """
    Load one header-free CSV file and extract a spectral vector.

    Supported formats:
        1. One column with expected_length rows
        2. One row with expected_length columns
        3. Multiple columns with expected_length rows, where the last column is used
        4. Any shape whose flattened size equals expected_length
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        raise ValueError(f"No numeric data found in file: {file_path}")

    if df.shape[0] == expected_length and df.shape[1] == 1:
        signal = df.iloc[:, 0].to_numpy(dtype=np.float32)

    elif df.shape[0] == 1 and df.shape[1] == expected_length:
        signal = df.iloc[0, :].to_numpy(dtype=np.float32)

    elif df.shape[0] == expected_length:
        signal = df.iloc[:, -1].to_numpy(dtype=np.float32)

    elif df.size == expected_length:
        signal = df.to_numpy(dtype=np.float32).flatten()

    else:
        raise ValueError(
            f"Expected {expected_length} data points, but got "
            f"{df.shape[0]} rows and {df.shape[1]} columns in file: {file_path}"
        )

    if len(signal) != expected_length:
        raise ValueError(
            f"Expected {expected_length} points, but got {len(signal)} points "
            f"in file: {file_path}"
        )

    return signal


# ============================================================
# Dataset
# ============================================================

class SpectralDataset(Dataset):
    """
    Dataset for 1D spectral classification.

    Expected label CSV columns:
        required:
            label

        recommended:
            relative_path

        optional:
            filename

    The 'relative_path' column is preferred because it avoids exposing
    local absolute paths and is suitable for GitHub repositories.
    """

    def __init__(
        self,
        label_df: pd.DataFrame,
        dataset_dir: Path,
        input_length: int = 2000,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.label_df = label_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.input_length = input_length
        self.mean = mean
        self.std = std

        self.file_index = None
        self.features, self.labels = self._load_all_samples()

    def _resolve_file_path(self, row: pd.Series) -> Path:
        """
        Resolve the CSV file path.

        Priority:
            1. relative_path under dataset_dir
            2. filename searched recursively under dataset_dir
        """
        if "relative_path" in row.index:
            path_value = row["relative_path"]

            if pd.notna(path_value):
                file_path = self.dataset_dir / str(path_value)

                if file_path.exists():
                    return file_path

        if "filename" in row.index:
            filename = str(row["filename"])

            if self.file_index is None:
                self.file_index = build_file_index(self.dataset_dir)

            if filename in self.file_index:
                return self.file_index[filename]

        raise FileNotFoundError(
            "Cannot locate file for this row. "
            "Please provide a valid 'relative_path' column in the label CSV."
        )

    def _load_all_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []

        for _, row in self.label_df.iterrows():
            file_path = self._resolve_file_path(row)

            signal = load_numeric_csv(
                file_path=file_path,
                expected_length=self.input_length,
            )

            label = int(row["label"])

            features.append(signal)
            labels.append(label)

        features = np.stack(features).astype(np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / (self.std + 1e-8)

        return features, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)

        return x, y


# ============================================================
# Class-wise split
# ============================================================

def split_by_class(
    label_df: pd.DataFrame,
    label_col: str = "label",
    val_per_class: int = 2,
    test_per_class: int = 7,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset class by class.

    For 30 spectra per class, the default split is:
        train: 21
        validation: 2
        test: 7
    """
    rng = np.random.default_rng(seed)

    train_parts = []
    val_parts = []
    test_parts = []

    for label, group in label_df.groupby(label_col):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n_samples = len(group)

        if n_samples <= val_per_class + test_per_class:
            raise ValueError(
                f"Class {label} has only {n_samples} samples, which is not enough "
                f"for {val_per_class} validation and {test_per_class} test samples."
            )

        indices = np.arange(n_samples)
        rng.shuffle(indices)

        test_indices = indices[:test_per_class]
        val_indices = indices[test_per_class:test_per_class + val_per_class]
        train_indices = indices[test_per_class + val_per_class:]

        test_parts.append(group.iloc[test_indices])
        val_parts.append(group.iloc[val_indices])
        train_parts.append(group.iloc[train_indices])

    train_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    test_df = (
        pd.concat(test_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df


# ============================================================
# CNN model
# ============================================================

class SpectralCNN4x4(nn.Module):
    """
    CNN for 2000-point 1D spectral classification.

    Network structure:
        Conv layer:
            Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=4)

        Fully connected layers:
            FC1: flatten -> 128
            FC2: 128 -> 64
            FC3: 64 -> num_classes

    Input:
        [batch_size, 2000]

    After convolution:
        [batch_size, 4, 500]

    After flatten:
        [batch_size, 2000]
    """

    def __init__(self, input_length: int = 2000, num_classes: int = 27):
        super().__init__()

        self.input_length = input_length

        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=4,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_output = self.conv_layer(dummy_input)
            flatten_dim = dummy_output.flatten(start_dim=1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.flatten(start_dim=1)
        logits = self.fc_layers(x)

        return logits


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * x.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return (
        avg_loss,
        acc,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ============================================================
# Plot utilities
# ============================================================

def plot_training_history(history_df: pd.DataFrame, output_dir: Path) -> None:
    """Save training loss and accuracy curves."""
    output_dir = Path(output_dir)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 4x4 CNN classifier for 1D spectral data."
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the dataset directory.",
    )

    parser.add_argument(
        "--label_csv",
        type=Path,
        required=True,
        help="Path to the label CSV file.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/cnn4x4_classification"),
        help="Directory for saving outputs.",
    )

    parser.add_argument(
        "--input_length",
        type=int,
        default=2000,
        help="Number of spectral points in each sample.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam optimizer.",
    )

    parser.add_argument(
        "--val_per_class",
        type=int,
        default=2,
        help="Number of validation samples per class.",
    )

    parser.add_argument(
        "--test_per_class",
        type=int,
        default=7,
        help="Number of test samples per class.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CNN 4x4 Spectral Classification")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Label CSV: {args.label_csv}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    label_df = pd.read_csv(args.label_csv)

    if "label" not in label_df.columns:
        raise ValueError("The label CSV must contain a 'label' column.")

    label_df["label"] = label_df["label"].astype(int)

    num_classes = label_df["label"].nunique()

    print(f"Total samples: {len(label_df)}")
    print(f"Number of classes: {num_classes}")
    print("Samples per class:")
    print(label_df["label"].value_counts().sort_index())
    print("=" * 70)

    train_df, val_df, test_df = split_by_class(
        label_df=label_df,
        label_col="label",
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print("=" * 70)

    train_df.to_csv(args.output_dir / "train_split.csv", index=False)
    val_df.to_csv(args.output_dir / "val_split.csv", index=False)
    test_df.to_csv(args.output_dir / "test_split.csv", index=False)

    temp_train_dataset = SpectralDataset(
        label_df=train_df,
        dataset_dir=args.dataset_dir,
        input_length=args.input_length,
    )

    train_features = temp_train_dataset.features
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)

    train_dataset = SpectralDataset(
        label_df=train_df,
        dataset_dir=args.dataset_dir,
        input_length=args.input_length,
        mean=mean,
        std=std,
    )

    val_dataset = SpectralDataset(
        label_df=val_df,
        dataset_dir=args.dataset_dir,
        input_length=args.input_length,
        mean=mean,
        std=std,
    )

    test_dataset = SpectralDataset(
        label_df=test_df,
        dataset_dir=args.dataset_dir,
        input_length=args.input_length,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = SpectralCNN4x4(
        input_length=args.input_length,
        num_classes=num_classes,
    ).to(device)

    print(model)
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_path = args.output_dir / "best_cnn4x4_classifier.pth"

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }
        )

        save_best = False

        if val_acc > best_val_acc:
            save_best = True

        elif val_acc == best_val_acc and val_loss < best_val_loss:
            save_best = True

        if save_best:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_length": int(args.input_length),
                    "num_classes": int(num_classes),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "best_val_acc": float(best_val_acc),
                    "best_val_loss": float(best_val_loss),
                    "seed": int(args.seed),
                },
                best_model_path,
            )

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.output_dir / "training_history.csv", index=False)

    plot_training_history(history_df, args.output_dir)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n" + "=" * 70)
    print("Final Test Results")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    report = classification_report(
        test_labels,
        test_preds,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(test_labels, test_preds)

    print("\nClassification Report")
    print(report)

    prediction_df = pd.DataFrame(
        {
            "true_label": test_labels,
            "predicted_label": test_preds,
        }
    )

    prediction_df.to_csv(args.output_dir / "test_predictions.csv", index=False)

    prob_columns = [f"class_{i}_prob" for i in range(num_classes)]
    prob_df = pd.DataFrame(test_probs, columns=prob_columns)
    prob_df.insert(0, "true_label", test_labels)
    prob_df.insert(1, "predicted_label", test_preds)

    prob_df.to_csv(
        args.output_dir / "test_prediction_probabilities.csv",
        index=False,
    )

    pd.DataFrame(cm).to_csv(
        args.output_dir / "confusion_matrix.csv",
        index=False,
    )

    with open(args.output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    metadata = {
        "model": "SpectralCNN4x4",
        "network_structure": "One convolution layer followed by three fully connected layers",
        "input_length": int(args.input_length),
        "first_layer": "Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=4)",
        "num_classes": int(num_classes),
        "total_samples": int(len(label_df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }

    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
