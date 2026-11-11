from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Use Times New Roman for all plot text, including math labels.
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.cal": "Times New Roman:italic",
    "mathtext.sf": "Times New Roman",
    "mathtext.tt": "Times New Roman",
    "axes.unicode_minus": False,
})
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    DEFAULT_CSV_PATH,
    denormalize,
    load_acoustic_data,
    prepare_surrogate_data,
)


class ForwardSurrogate(nn.Module):
    """Map [geometry + query point] to one frequency value."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden_dim), nn.SiLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    loss_fn = nn.MSELoss(reduction="sum")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_count += len(x)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_numpy(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
        pred = model(batch).detach().cpu().numpy()
        outputs.append(pred)
    return np.vstack(outputs).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def plot_loss(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss, normalized frequency")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train forward surrogate for acoustic band structure.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--out-dir", default="outputs/surrogate")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_acoustic_data(
        csv_path=args.csv,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    prepared = prepare_surrogate_data(data, normalize_data=True)

    x_train = prepared["x_train_norm"]
    y_train = prepared["y_train_norm"]
    x_val = prepared["x_val_norm"]
    y_val = prepared["y_val_norm"]

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    device = get_device()
    model = ForwardSurrogate(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best_val = float("inf")
    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train pairs: {len(x_train):,} | Validation pairs: {len(x_val):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, optimizer=None)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": int(x_train.shape[1]),
                        "hidden_dim": args.hidden_dim,
                        "depth": args.depth,
                    },
                    "metadata": data["metadata"],
                    "x_scaler": prepared["x_scaler"],
                    "y_scaler": prepared["y_scaler"],
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                out_dir / "surrogate_best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "surrogate_last_state_dict.pt")

    loss_array = np.asarray([[h["epoch"], h["train_loss"], h["val_loss"]] for h in history], dtype=np.float64)
    np.savetxt(out_dir / "loss_curve.csv", loss_array, delimiter=",", header="epoch,train_loss,val_loss", comments="")
    plot_loss(history, out_dir / "loss_curve.png")

    checkpoint = torch.load(out_dir / "surrogate_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_val_pred_norm = predict_numpy(model, x_val, device, args.eval_batch_size)
    y_val_pred = denormalize(y_val_pred_norm, prepared["y_scaler"])
    y_val_raw = data["val_rows"][["frequency"]].to_numpy(dtype=np.float32)
    metrics = regression_metrics(y_val_raw, y_val_pred)
    metrics["best_val_loss_norm"] = float(best_val)
    metrics["n_train_pairs"] = int(len(x_train))
    metrics["n_val_pairs"] = int(len(x_val))

    save_json(metrics, out_dir / "metrics.json")
    save_json(data["metadata"], out_dir / "metadata.json")

    print("Validation metrics on raw frequency:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
