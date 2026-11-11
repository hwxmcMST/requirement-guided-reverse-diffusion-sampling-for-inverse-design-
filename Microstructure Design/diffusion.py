from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data_utils import load_torch_checkpoint, atomic_torch_save, make_diffusion_dataloaders, save_image_grid


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    emb = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=1)
    return F.pad(emb, (0, dim % 2))


def norm(channels: int) -> nn.GroupNorm:
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def make_position_grid() -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, 28)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    rr = torch.sqrt(xx.square() + yy.square()) / math.sqrt(2.0)
    return torch.stack([xx, yy, rr], dim=0).unsqueeze(0)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = norm(out_ch)
        self.norm2 = norm(out_ch)
        self.time = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(time_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return (h + residual) / math.sqrt(2.0)


class GlobalBottleneck(nn.Module):
    def __init__(self, channels: int, size: int = 7):
        super().__init__()
        flat_dim = channels * size * size
        bottleneck_dim = max(256, channels * 4)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, flat_dim),
            nn.Unflatten(1, (channels, size, size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.net(x)) / math.sqrt(2.0)


class NoisePredictor(nn.Module):
    def __init__(self, hidden: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU())
        self.register_buffer("position_grid", make_position_grid())
        self.in_conv = nn.Conv2d(4, hidden, 3, padding=1)

        self.down1 = Block(hidden, hidden, time_dim)
        self.to_14 = nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1)
        self.down2 = Block(hidden * 2, hidden * 2, time_dim)
        self.to_7 = nn.Conv2d(hidden * 2, hidden * 4, 4, stride=2, padding=1)
        self.mid = Block(hidden * 4, hidden * 4, time_dim)
        self.global_mid = GlobalBottleneck(hidden * 4)

        self.up14 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1)
        self.up1 = Block(hidden * 4, hidden * 2, time_dim)
        self.up28 = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.up2 = Block(hidden * 2, hidden, time_dim)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, n_steps: int) -> torch.Tensor:
        time_emb = self.time_mlp(timestep_embedding(t.float(), self.time_dim))
        pos = self.position_grid.expand(x.size(0), -1, -1, -1)
        x = torch.cat([x, pos], dim=1)
        x = self.in_conv(x)

        s1 = self.down1(x, time_emb)
        s2 = self.down2(self.to_14(s1), time_emb)
        x = self.mid(self.to_7(s2), time_emb)
        x = self.global_mid(x)
        x = self.up1(torch.cat([self.up14(x), s2], dim=1), time_emb)
        x = self.up2(torch.cat([self.up28(x), s1], dim=1), time_emb)
        return self.out(x)


def make_schedule(n_steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "posterior_var": betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
    }


def extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return values.gather(0, t).view(t.shape[0], 1, 1, 1)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: dict) -> torch.Tensor:
    return extract(schedule["sqrt_alpha_bar"], t, x0) * x0 + extract(
        schedule["sqrt_one_minus_alpha_bar"], t, x0
    ) * noise


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def train_epoch(model, loader, optimizer, schedule, n_steps, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for x0 in loader:
        x0 = x0.to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, n_steps, (x0.size(0),), device=device)
        xt = q_sample(x0, t, noise, schedule)
        loss = F.mse_loss(model(xt, t, n_steps), noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item() * x0.size(0)
        count += x0.size(0)
    return total / count


@torch.no_grad()
def eval_loss(model, loader, schedule, n_steps, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x0 in loader:
        x0 = x0.to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, n_steps, (x0.size(0),), device=device)
        xt = q_sample(x0, t, noise, schedule)
        loss = F.mse_loss(model(xt, t, n_steps), noise)
        total += loss.item() * x0.size(0)
        count += x0.size(0)
    return total / count


@torch.no_grad()
def sample(
    model,
    n_samples: int,
    schedule: dict,
    n_steps: int,
    device,
    return_zero_one: bool = True,
) -> np.ndarray:
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    for step in reversed(range(n_steps)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t, n_steps)
        beta = extract(schedule["betas"], t, x)
        alpha = extract(schedule["alphas"], t, x)
        sqrt_omab = extract(schedule["sqrt_one_minus_alpha_bar"], t, x)
        mean = (x - beta * pred_noise / sqrt_omab) / torch.sqrt(alpha)
        if step > 0:
            x = mean + torch.sqrt(extract(schedule["posterior_var"], t, x)) * torch.randn_like(x)
        else:
            x = mean

    x = to_zero_one(x) if return_zero_one else x.clamp(-1.0, 1.0)
    return x.cpu().numpy().astype(np.float32)


def load_diffusion_model(path: str | Path, device) -> tuple[NoisePredictor, dict, int, dict]:
    checkpoint = load_torch_checkpoint(path, device)

    args = checkpoint["args"]
    n_steps = int(args["n_steps"])
    model = NoisePredictor(hidden=int(args["hidden"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, make_schedule(n_steps, device), n_steps, checkpoint


def plot_loss(history: list[dict], path: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise prediction MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.n_samples, args.n_steps, args.hidden) < 1:
        parser.error("epochs, batch-size, n-samples, n-steps and hidden must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = make_diffusion_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )

    model = NoisePredictor(args.hidden).to(device)
    schedule = make_schedule(args.n_steps, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []
    ckpt_path = out_dir / "diffusion_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, schedule, args.n_steps, device)
        val_loss = eval_loss(model, val_loader, schedule, args.n_steps, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val,
                    "model_version": "v2_pos_res_global",
                },
                ckpt_path,
            )
        print(f"epoch {epoch:04d} train={train_loss:.6g} val={val_loss:.6g}")

    model, schedule, n_steps, _ = load_diffusion_model(ckpt_path, device)
    samples = sample(model, args.n_samples, schedule, n_steps, device, return_zero_one=True)
    np.save(out_dir / "generated_samples.npy", samples)
    save_image_grid(samples, out_dir / "generated_samples.png", args.n_samples)
    plot_loss(history, out_dir / "loss_curve.png")

    metrics = {
        "model_version": "v2_pos_res_global",
        "best_val_loss": float(best_val),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_steps": args.n_steps,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
