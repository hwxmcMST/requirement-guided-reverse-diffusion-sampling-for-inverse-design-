from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_utils import load_torch_checkpoint, DISPLACEMENTS, CurveScaler, atomic_torch_save, make_surrogate_dataloaders, save_metadata


class EnergySurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 13),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, curves in loader:
        images = images.to(device)
        curves = curves.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss = loss_fn(pred, curves)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for images, curves in loader:
        images = images.to(device)
        curves = curves.to(device)
        loss = loss_fn(model(images), curves)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_normalized(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []

    for images, curves in loader:
        pred = model(images.to(device)).detach().cpu().numpy()
        pred_list.append(pred)
        true_list.append(curves.numpy())

    return np.concatenate(pred_list, axis=0), np.concatenate(true_list, axis=0)


def regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_raw(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    scaler: CurveScaler,
    device: torch.device,
) -> dict[str, float | np.ndarray]:
    pred_norm, true_norm = predict_normalized(model, loader, device)
    pred = scaler.inverse_transform(pred_norm)
    true = scaler.inverse_transform(true_norm)
    metrics = regression_metrics(pred, true)
    metrics["pred"] = pred
    metrics["true"] = true
    return metrics


def save_loss_curve(history: list[dict[str, float]], path: str | Path) -> None:
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

    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss, normalized energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_prediction_plot(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    path: str | Path,
    n_examples: int = 6,
) -> None:
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

    n_examples = min(n_examples, len(y_true))
    fig, axes = plt.subplots(n_examples, 1, figsize=(6.0, 2.2 * n_examples), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.plot(DISPLACEMENTS, y_true[i], "o-", label="ground truth")
        ax.plot(DISPLACEMENTS, y_pred[i], "s--", label="prediction")
        ax.set_ylabel("Energy")
        ax.set_title(f"test sample {i}")
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Displacement")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    scaler: CurveScaler,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "curve_scaler": {"mean": scaler.mean, "std": scaler.std},
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[EnergySurrogate, CurveScaler, dict]:
    checkpoint = load_torch_checkpoint(path, device)
    model = EnergySurrogate()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    scaler = CurveScaler(**checkpoint["curve_scaler"])
    return model, scaler, checkpoint


def write_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/surrogate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-factor", type=float, default=0.9)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch-size must be positive")

    set_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, scaler = make_surrogate_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_test=args.max_test,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = EnergySurrogate().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    checkpoint_path = out_dir / "surrogate_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        if val_loss < best_val:
            best_val = val_loss
            val_metrics_raw = evaluate_raw(model, val_loader, scaler, device)
            best_metrics = {
                key: float(val_metrics_raw[key])
                for key in ["mse", "rmse", "mae", "r2"]
            }
            best_metrics["best_val_loss_norm"] = float(best_val)
            save_checkpoint(checkpoint_path, model, scaler, args, best_metrics)

        print(
            f"epoch {epoch:04d} train={train_loss:.6g} val={val_loss:.6g} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}"
        )

    model, scaler, _ = load_checkpoint(checkpoint_path, device)
    test_eval = evaluate_raw(model, test_loader, scaler, device)
    test_metrics = {
        key: float(test_eval[key])
        for key in ["mse", "rmse", "mae", "r2"]
    }
    test_metrics["best_val_loss_norm"] = float(best_val)
    test_metrics["n_train"] = len(train_loader.dataset)
    test_metrics["n_val"] = len(val_loader.dataset)
    test_metrics["n_test"] = len(test_loader.dataset)

    np.savetxt(
        out_dir / "loss_curve.csv",
        np.array([[h["epoch"], h["train_loss"], h["val_loss"], h["lr"]] for h in history]),
        delimiter=",",
        header="epoch,train_loss,val_loss,lr",
        comments="",
    )
    write_json(out_dir / "metrics.json", test_metrics)
    save_metadata(out_dir / "metadata.json", scaler)
    save_loss_curve(history, out_dir / "loss_curve.png")
    save_prediction_plot(
        test_eval["pred"], test_eval["true"], out_dir / "test_curve_examples.png"
    )

    print("Validation metrics at best checkpoint:")
    print(json.dumps(best_metrics, indent=2))
    print("Test metrics on raw energy curve:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
