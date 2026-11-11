from __future__ import annotations

import argparse
import json
import shutil
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
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    DEFAULT_CSV_PATH,
    GEOMETRY_COLS,
    check_design_validity,
    denormalize,
    load_acoustic_data,
    prepare_diffusion_data,
    project_designs,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class NoisePredictor(nn.Module):
    """Predict Gaussian noise added to a normalized 3-variable geometry vector."""

    def __init__(self, design_dim: int = 3, time_dim: int = 64, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        self.time_dim = time_dim

        layers: list[nn.Module] = []
        dim = design_dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden_dim), nn.SiLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, design_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, timestep_embedding(t, self.time_dim)], dim=1))


class DiffusionSchedule:
    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    loss_fn = nn.MSELoss(reduction="sum")

    for (x0,) in loader:
        x0 = x0.to(device)
        t = torch.randint(0, schedule.steps, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        xt = schedule.q_sample(x0, t, noise)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred_noise = model(xt, t)
        loss = loss_fn(pred_noise, noise)

        if training:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_count += len(x0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def sample_normalized_designs(
    model: nn.Module,
    n_samples: int,
    schedule: DiffusionSchedule,
    device: torch.device,
    design_dim: int = 3,
) -> np.ndarray:
    model.eval()
    x = torch.randn(n_samples, design_dim, device=device)

    for step in reversed(range(schedule.steps)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]

        pred_noise = model(x, t)
        x = (x - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
        if step > 0:
            x = x + torch.sqrt(beta) * torch.randn_like(x)

    return x.detach().cpu().numpy().astype(np.float32)


def decode_samples(
    samples_norm: np.ndarray,
    scaler: dict[str, list[float]],
    bounds: dict[str, list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(denormalize(samples_norm, scaler), columns=GEOMETRY_COLS)
    raw_checked = check_design_validity(raw, bounds)
    raw_checked.insert(0, "sample_id", np.arange(len(raw_checked), dtype=int))

    projected = project_designs(raw, bounds)
    checked = check_design_validity(projected, bounds)
    checked.insert(0, "sample_id", np.arange(len(checked), dtype=int))
    return raw_checked, checked


def smooth_loss_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    smoothed = frame.copy()
    window = max(1, min(int(window), len(smoothed)))
    for column in ["train_loss", "val_loss"]:
        smoothed[f"{column}_smooth"] = smoothed[column].rolling(
            window=window,
            center=True,
            min_periods=1,
        ).mean()
    return smoothed


def plot_loss(history: list[dict[str, float]], path: Path, smooth_window: int = 51) -> pd.DataFrame:
    frame = smooth_loss_frame(pd.DataFrame(history), smooth_window)
    plt.figure(figsize=(7, 4.5))
    plt.plot(frame["epoch"], frame["train_loss_smooth"], label="train")
    plt.plot(frame["epoch"], frame["val_loss_smooth"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel(f"Noise prediction MSE, rolling mean ({smooth_window})")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    return frame


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an unconditional diffusion model for acoustic geometries.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--out-dir", default="outputs/diffusion")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_acoustic_data(args.csv, args.train_size, args.val_size, args.seed)
    prepared = prepare_diffusion_data(data, normalize_data=True)
    x_train = prepared["x_train_norm"]
    x_val = prepared["x_val_norm"]

    device = get_device()
    train_loader = make_loader(x_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, args.batch_size, shuffle=False)
    schedule = DiffusionSchedule(args.diffusion_steps, args.beta_start, args.beta_end, device)
    model = NoisePredictor(
        design_dim=x_train.shape[1],
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    print(f"Device: {device}")
    print(f"Train designs: {len(x_train):,} | Validation designs: {len(x_val):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, schedule, device, optimizer, args.grad_clip)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, schedule, device, optimizer=None)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "design_dim": int(x_train.shape[1]),
                        "time_dim": args.time_dim,
                        "hidden_dim": args.hidden_dim,
                        "depth": args.depth,
                    },
                    "diffusion_config": {
                        "steps": args.diffusion_steps,
                        "beta_start": args.beta_start,
                        "beta_end": args.beta_end,
                    },
                    "metadata": data["metadata"],
                    "geometry_scaler": prepared["scaler"],
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                out_dir / "diffusion_best.pt",
            )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "diffusion_last_state_dict.pt")
    pd.DataFrame(history).to_csv(out_dir / "loss_curve.csv", index=False)
    smoothed_loss = plot_loss(history, out_dir / "loss_curve.png", args.smooth_window)
    shutil.copyfile(out_dir / "loss_curve.png", out_dir / "loss_curve_smoothed.png")
    smoothed_loss.to_csv(out_dir / "loss_curve_smoothed.csv", index=False)

    checkpoint = torch.load(out_dir / "diffusion_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    samples_norm = sample_normalized_designs(model, args.n_samples, schedule, device, x_train.shape[1])
    raw_generated, projected_generated = decode_samples(
        samples_norm,
        prepared["scaler"],
        data["metadata"]["geometry_bounds"],  # type: ignore[index]
    )

    raw_generated.to_csv(out_dir / "generated_designs_raw.csv", index=False)
    projected_generated.to_csv(out_dir / "generated_designs_projected.csv", index=False)
    np.save(out_dir / "generated_designs_norm.npy", samples_norm)

    summary = {
        "best_val_loss": float(best_val),
        "n_train_designs": int(len(x_train)),
        "n_val_designs": int(len(x_val)),
        "n_generated": int(len(raw_generated)),
        "n_valid_raw": int(raw_generated["is_valid"].sum()),
        "valid_rate_raw": float(raw_generated["is_valid"].mean()),
        "n_valid_after_projection": int(projected_generated["is_valid"].sum()),
        "valid_rate_after_projection": float(projected_generated["is_valid"].mean()),
        "note": "Unconditional geometry diffusion only; no surrogate prediction or requirement score.",
    }
    save_json({"summary": summary, "metadata": data["metadata"]}, out_dir / "metadata.json")

    print("Generation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
