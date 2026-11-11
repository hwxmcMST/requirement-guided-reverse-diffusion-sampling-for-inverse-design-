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
import pandas as pd
import torch

from data_utils import (
    DEFAULT_CSV_PATH,
    GEOMETRY_COLS,
    denormalize,
    load_acoustic_data,
    prepare_surrogate_data,
)
from diffusion import DiffusionSchedule, NoisePredictor, decode_samples, get_device, sample_normalized_designs
from surrogate import ForwardSurrogate, predict_numpy, regression_metrics


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_diffusion(checkpoint_path: str | Path, device: torch.device) -> tuple[NoisePredictor, DiffusionSchedule, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NoisePredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cfg = checkpoint["diffusion_config"]
    schedule = DiffusionSchedule(cfg["steps"], cfg["beta_start"], cfg["beta_end"], device)
    return model, schedule, checkpoint


def load_surrogate(checkpoint_path: str | Path, device: torch.device) -> tuple[ForwardSurrogate, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ForwardSurrogate(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def geometry_summary(test: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in [("test", test), ("generated_projected", generated)]:
        for col in GEOMETRY_COLS:
            values = frame[col].to_numpy(dtype=float)
            rows.append(
                {
                    "group": name,
                    "feature": col,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "q75": float(np.quantile(values, 0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def plot_geometry_histograms(test: pd.DataFrame, generated: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, GEOMETRY_COLS):
        ax.hist(test[col], bins=18, density=True, alpha=0.55, label="test")
        ax.hist(generated[col], bins=24, density=True, alpha=0.45, label="generated")
        ax.set_title(col)
        ax.set_xlabel("radius")
        ax.set_ylabel("density")
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_geometry_scatter(test: pd.DataFrame, generated: pd.DataFrame, path: Path, max_generated: int = 1000) -> None:
    if len(generated) > max_generated:
        generated = generated.sample(max_generated, random_state=7)

    pairs = [
        ("strut_radius", "center_mass_radius"),
        ("strut_radius", "corner_mass_radius"),
        ("center_mass_radius", "corner_mass_radius"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (x_col, y_col) in zip(axes, pairs):
        ax.scatter(generated[x_col], generated[y_col], s=12, alpha=0.35, label="generated")
        ax.scatter(test[x_col], test[y_col], s=24, alpha=0.85, label="test")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_surrogate_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    ax.plot([low, high], [low, high], color="black", linewidth=1.2)
    ax.set_xlabel("Ground truth frequency")
    ax.set_ylabel("Predicted frequency")
    ax.set_title("Forward surrogate on test rows")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate diffusion samples and forward surrogate test predictions.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--diffusion-checkpoint", default="outputs/diffusion/diffusion_best.pt")
    parser.add_argument("--surrogate-checkpoint", default="outputs/surrogate/surrogate_best.pt")
    parser.add_argument("--out-dir", default="outputs/validation")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    data = load_acoustic_data(args.csv, args.train_size, args.val_size, args.seed)
    test_designs = data["val_designs"][GEOMETRY_COLS].copy()

    diffusion_model, diffusion_schedule, diffusion_ckpt = load_diffusion(args.diffusion_checkpoint, device)
    samples_norm = sample_normalized_designs(
        diffusion_model,
        args.n_samples,
        diffusion_schedule,
        device,
        design_dim=diffusion_ckpt["model_config"]["design_dim"],
    )
    raw_generated, projected_generated = decode_samples(
        samples_norm,
        diffusion_ckpt["geometry_scaler"],
        data["metadata"]["geometry_bounds"],  # type: ignore[index]
    )
    raw_generated.to_csv(out_dir / "generated_designs_raw.csv", index=False)
    projected_generated.to_csv(out_dir / "generated_designs_projected.csv", index=False)

    projected_designs = projected_generated[GEOMETRY_COLS].copy()
    geometry_summary(test_designs, projected_designs).to_csv(out_dir / "geometry_distribution_summary.csv", index=False)
    plot_geometry_histograms(test_designs, projected_designs, out_dir / "geometry_histograms.png")
    plot_geometry_scatter(test_designs, projected_designs, out_dir / "geometry_pairwise_scatter.png")

    surrogate_model, surrogate_ckpt = load_surrogate(args.surrogate_checkpoint, device)
    prepared = prepare_surrogate_data(data, normalize_data=True)
    x_test = prepared["x_val_norm"]
    y_test_raw = data["val_rows"][["frequency"]].to_numpy(dtype=np.float32)
    y_pred_norm = predict_numpy(surrogate_model, x_test, device, args.eval_batch_size)
    y_pred_raw = denormalize(y_pred_norm, surrogate_ckpt.get("y_scaler", prepared["y_scaler"]))

    metrics = regression_metrics(y_test_raw, y_pred_raw)
    metrics["n_test_rows"] = int(len(y_test_raw))
    save_json(metrics, out_dir / "surrogate_test_metrics.json")
    plot_surrogate_predictions(y_test_raw, y_pred_raw, out_dir / "surrogate_pred_vs_ground_truth.png")

    prediction_table = data["val_rows"].copy()
    prediction_table["frequency_pred"] = y_pred_raw.reshape(-1)
    prediction_table["abs_error"] = np.abs(prediction_table["frequency_pred"] - prediction_table["frequency"])
    prediction_table.to_csv(out_dir / "surrogate_test_predictions.csv", index=False)

    summary = {
        "diffusion": {
            "n_generated": int(len(raw_generated)),
            "n_raw_valid": int(raw_generated["is_valid"].sum()),
            "raw_valid_rate": float(raw_generated["is_valid"].mean()),
            "distribution_outputs": [
                "geometry_distribution_summary.csv",
                "geometry_histograms.png",
                "geometry_pairwise_scatter.png",
            ],
        },
        "surrogate": metrics,
    }
    save_json(summary, out_dir / "validation_summary.json")

    print("Validation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
