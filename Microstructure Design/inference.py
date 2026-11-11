from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data_utils import load_torch_checkpoint, CurveScaler, DISPLACEMENTS, energy_path, load_energy_curves, save_image_grid
from diffusion import extract, load_diffusion_model, sample, to_zero_one
from requirement import (
    Requirement,
    early_max_energy,
    evaluate_curves,
    final_energy,
    hard_satisfaction,
    parse_requirement,
)
from surrogate import EnergySurrogate


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_surrogate(path: str | Path, device: torch.device) -> tuple[EnergySurrogate, CurveScaler]:
    ckpt = load_torch_checkpoint(path, device)
    model = EnergySurrogate().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, CurveScaler(**ckpt["curve_scaler"])


@torch.no_grad()
def predict_curves(model, scaler: CurveScaler, images: np.ndarray, device, batch_size: int) -> np.ndarray:
    if len(images) < 1 or batch_size < 1:
        raise ValueError("Prediction requires nonempty images and a positive batch size.")
    pred = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i : i + batch_size]).float().to(device)
        pred.append(model(x).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(pred, axis=0))


def requirement_distance(curves: np.ndarray, req: Requirement) -> np.ndarray:
    if req.kind == "final_energy_range":
        y = final_energy(curves)
        return np.maximum(req.psi_min - y, 0.0) + np.maximum(y - req.psi_max, 0.0)
    if req.kind == "low_early_energy":
        y = early_max_energy(curves, req.early_u_max)
        return np.maximum(y - req.c1, 0.0)
    raise ValueError(f"Unknown requirement kind: {req.kind}")


@torch.no_grad()
def reverse_step(model, x: torch.Tensor, step: int, schedule: dict, n_steps: int) -> torch.Tensor:
    t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
    eps = model(x, t, n_steps)
    beta = extract(schedule["betas"], t, x)
    alpha = extract(schedule["alphas"], t, x)
    sqrt_omab = extract(schedule["sqrt_one_minus_alpha_bar"], t, x)
    mean = (x - beta * eps / sqrt_omab) / torch.sqrt(alpha)
    if step == 0:
        return mean
    var = extract(schedule["posterior_var"], t, x)
    return mean + torch.sqrt(var) * torch.randn_like(x)


@torch.no_grad()
def estimate_x0(model, x: torch.Tensor, step: int, schedule: dict, n_steps: int) -> torch.Tensor:
    if step < 0:
        return x.clamp(-1.0, 1.0)
    t = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
    eps = model(x, t, n_steps)
    x0 = (x - extract(schedule["sqrt_one_minus_alpha_bar"], t, x) * eps) / extract(
        schedule["sqrt_alpha_bar"], t, x
    )
    return x0.clamp(-1.0, 1.0)


@torch.no_grad()
def guided_sample(
    diffusion,
    surrogate,
    scaler: CurveScaler,
    req: Requirement,
    n_samples: int,
    k: int,
    schedule: dict,
    n_steps: int,
    device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = torch.randn(n_samples, 1, 28, 28, device=device)

    for step in reversed(range(n_steps)):
        candidates = reverse_step(diffusion, x.repeat_interleave(k, 0), step, schedule, n_steps)
        x0 = to_zero_one(estimate_x0(diffusion, candidates, step - 1, schedule, n_steps))
        curves = predict_curves(surrogate, scaler, x0.cpu().numpy().astype(np.float32), device, batch_size)
        feasible = hard_satisfaction(curves, req).reshape(n_samples, k)
        distance = requirement_distance(curves, req).reshape(n_samples, k)

        chosen = []
        for i in range(n_samples):
            ok = np.flatnonzero(feasible[i])
            chosen.append(int(rng.choice(ok)) if len(ok) else int(distance[i].argmin()))
        idx = torch.arange(n_samples, device=device) * k + torch.tensor(chosen, device=device)
        x = candidates[idx]

    return to_zero_one(x).cpu().numpy().astype(np.float32)


def pick_examples(curves: np.ndarray, req: Requirement, prefer_feasible: bool, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    flags = hard_satisfaction(curves, req).astype(bool)
    pool = np.flatnonzero(flags if prefer_feasible else ~flags)
    picked = rng.choice(pool, min(n, len(pool)), replace=False).tolist() if len(pool) else []
    rest = np.setdiff1d(np.arange(len(curves)), np.asarray(picked), assume_unique=False)
    if len(picked) < n and len(rest):
        picked += rng.choice(rest, min(n - len(picked), len(rest)), replace=False).tolist()
    return [int(i) for i in picked]


def plot_examples(unguided: np.ndarray, guided: np.ndarray, req: Requirement, path: Path, seed: int) -> dict:
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

    indices = {
        "unguided": pick_examples(unguided, req, False, 3, seed),
        "guided": pick_examples(guided, req, True, 3, seed + 1),
    }
    rows = [("unguided", unguided), ("guided", guided)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2), sharex=True)

    for r, (name, curves) in enumerate(rows):
        flags = hard_satisfaction(curves, req)
        for c, idx in enumerate(indices[name]):
            ax = axes[r, c]
            ax.plot(DISPLACEMENTS, curves[idx], "o-", color=f"tab:{'blue' if r == 0 else 'orange'}")
            if req.kind == "final_energy_range":
                ax.axhspan(req.psi_min, req.psi_max, color="tab:green", alpha=0.12)
            else:
                ax.axhline(req.c1, color="tab:green", linestyle="--", linewidth=1.0)
            ax.set_title(f"{name} #{idx}\n{'satisfied' if flags[idx] else 'not satisfied'}", fontsize=10)
            ax.grid(alpha=0.2)
            if c == 0:
                ax.set_ylabel("Energy")
            if r == 1:
                ax.set_xlabel("Displacement")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return indices


def save_feasible_grid(images: np.ndarray, curves: np.ndarray, req: Requirement, path: Path) -> list[int]:
    flags = hard_satisfaction(curves, req).astype(bool)
    indices = np.flatnonzero(flags).astype(int).tolist()
    if indices:
        save_image_grid(images[flags], path, len(indices))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirement", choices=["R1", "R2"], required=True)
    parser.add_argument("--diffusion-checkpoint", default=str(Path(__file__).resolve().parent / "outputs/diffusion/diffusion_best.pt"))
    parser.add_argument("--surrogate-checkpoint", default=str(Path(__file__).resolve().parent / "outputs/surrogate/surrogate_best.pt"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs/inference"))
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--k-candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--psi-min", type=float, default=None)
    parser.add_argument("--psi-max", type=float, default=None)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--early-u-max", type=float, default=1.0)
    parser.add_argument("--r1-q-min", type=float, default=0.6)
    parser.add_argument("--r1-q-max", type=float, default=0.8)
    parser.add_argument("--r2-q", type=float, default=0.3)
    args = parser.parse_args()
    if min(args.n_samples, args.k_candidates, args.batch_size) < 1:
        parser.error("n-samples, k-candidates and batch-size must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.output_dir) / args.requirement
    out_dir.mkdir(parents=True, exist_ok=True)

    diffusion, schedule, n_steps, diffusion_ckpt = load_diffusion_model(args.diffusion_checkpoint, device)
    surrogate, scaler = load_surrogate(args.surrogate_checkpoint, device)
    req = parse_requirement(args, load_energy_curves(energy_path("train")))

    unguided_images = sample(diffusion, args.n_samples, schedule, n_steps, device, return_zero_one=True)
    guided_images = guided_sample(
        diffusion, surrogate, scaler, req, args.n_samples, args.k_candidates,
        schedule, n_steps, device, args.batch_size, args.seed
    )
    unguided_curves = predict_curves(surrogate, scaler, unguided_images, device, args.batch_size)
    guided_curves = predict_curves(surrogate, scaler, guided_images, device, args.batch_size)

    np.save(out_dir / "unguided_samples.npy", unguided_images)
    np.save(out_dir / "guided_samples.npy", guided_images)
    np.save(out_dir / "unguided_pred_curves.npy", unguided_curves)
    np.save(out_dir / "guided_pred_curves.npy", guided_curves)
    save_image_grid(unguided_images, out_dir / "unguided_samples.png", args.n_samples)
    save_image_grid(guided_images, out_dir / "guided_samples.png", args.n_samples)
    feasible_indices = {
        "unguided": save_feasible_grid(unguided_images, unguided_curves, req, out_dir / "unguided_feasible_samples.png"),
        "guided": save_feasible_grid(guided_images, guided_curves, req, out_dir / "guided_feasible_samples.png"),
    }
    examples = plot_examples(unguided_curves, guided_curves, req, out_dir / "predicted_curve_examples.png", args.seed)

    metrics = {
        "requirement": asdict(req),
        "diffusion_version": diffusion_ckpt.get("model_version"),
        "seed": args.seed,
        "diffusion_steps": n_steps,
        "n_samples": args.n_samples,
        "k_candidates": args.k_candidates,
        "feasible_indices": feasible_indices,
        "plotted_example_indices": examples,
        "unguided": evaluate_curves(unguided_curves, req),
        "guided": evaluate_curves(guided_curves, req),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
