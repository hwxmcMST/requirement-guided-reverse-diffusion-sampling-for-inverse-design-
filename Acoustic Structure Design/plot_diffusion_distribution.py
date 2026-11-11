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


DEFAULT_PROJECT_DIR = Path("/Users/lin.ke/Desktop/project/case_1_3")
GEOMETRY_COLS = ["strut_radius", "center_mass_radius", "corner_mass_radius"]
LABELS = {
    "strut_radius": r"$r_s$",
    "center_mass_radius": r"$r_c$",
    "corner_mass_radius": r"$r_m$",
}
COLORS = {
    "training": "#2b2b2b",
    "generated": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and diffusion-generated geometry distributions."
    )
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--train-designs", type=Path, default=None)
    parser.add_argument("--generated-designs", type=Path, default=None)
    parser.add_argument("--raw-generated-designs", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-generated", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_bounds(metadata_path: Path | None) -> dict[str, list[float]] | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "metadata" in payload:
        payload = payload["metadata"]
    return payload.get("geometry_bounds")


def sample_generated(generated: pd.DataFrame, max_generated: int, seed: int) -> pd.DataFrame:
    if len(generated) <= max_generated:
        return generated
    return generated.sample(max_generated, random_state=seed)


def save_summary(
    train: pd.DataFrame,
    generated: pd.DataFrame,
    raw_generated: pd.DataFrame | None,
    path: Path,
) -> None:
    rows = []
    for group, frame in [("training", train), ("generated", generated)]:
        for col in GEOMETRY_COLS:
            values = frame[col].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
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
    summary = pd.DataFrame(rows)
    summary.to_csv(path.with_suffix(".csv"), index=False)

    json_payload = {
        "n_training": int(len(train)),
        "n_generated": int(len(generated)),
    }
    if raw_generated is not None and "is_valid" in raw_generated:
        raw_valid_rate = float(raw_generated["is_valid"].astype(bool).mean())
        json_payload["raw_valid_rate_before_projection"] = raw_valid_rate
        json_payload["raw_valid_percent_before_projection"] = 100.0 * raw_valid_rate
    if "is_valid" in generated:
        projected_valid_rate = float(generated["is_valid"].astype(bool).mean())
        json_payload["generated_valid_rate"] = projected_valid_rate
        json_payload["generated_valid_percent"] = 100.0 * projected_valid_rate

    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)


def set_pair_limits(ax, x_col: str, y_col: str, bounds: dict[str, list[float]] | None) -> None:
    if bounds is None:
        return
    ax.set_xlim(bounds[x_col])
    ax.set_ylim(bounds[y_col])


def kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(grid)

    std = values.std(ddof=1) if len(values) > 1 else 0.0
    data_range = max(float(values.max() - values.min()), 1e-6)
    bandwidth = 1.06 * std * len(values) ** (-1.0 / 5.0)
    if not np.isfinite(bandwidth) or bandwidth <= 1e-8:
        bandwidth = data_range / 20.0

    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2.0 * np.pi)
    return density


def plot_density_curves(
    train: pd.DataFrame,
    generated: pd.DataFrame,
    bounds: dict[str, list[float]] | None,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    for ax, col in zip(axes, GEOMETRY_COLS):
        if bounds is not None:
            lo, hi = bounds[col]
        else:
            values = np.concatenate([train[col].to_numpy(dtype=float), generated[col].to_numpy(dtype=float)])
            lo, hi = float(values.min()), float(values.max())
            pad = 0.08 * max(hi - lo, 1e-6)
            lo, hi = lo - pad, hi + pad

        grid = np.linspace(lo, hi, 400)
        train_density = kde_1d(train[col].to_numpy(dtype=float), grid)
        generated_density = kde_1d(generated[col].to_numpy(dtype=float), grid)

        ax.fill_between(
            grid,
            train_density,
            color=COLORS["training"],
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(grid, train_density, color=COLORS["training"], lw=1.8, label="Training")
        ax.fill_between(
            grid,
            generated_density,
            color=COLORS["generated"],
            alpha=0.25,
            linewidth=0,
        )
        ax.plot(grid, generated_density, color=COLORS["generated"], lw=1.8, ls="--", label="Generated")
        ax.set_xlabel(LABELS[col])
        ax.set_ylabel("Density")
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.grid(True, color="0.92", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_3d_distribution(
    train: pd.DataFrame,
    generated: pd.DataFrame,
    bounds: dict[str, list[float]] | None,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        train["strut_radius"],
        train["center_mass_radius"],
        train["corner_mass_radius"],
        s=22,
        c="black",
        alpha=0.55,
        depthshade=False,
        label="Training",
    )
    ax.scatter(
        generated["strut_radius"],
        generated["center_mass_radius"],
        generated["corner_mass_radius"],
        s=28,
        c="tab:red",
        alpha=0.72,
        depthshade=False,
        label="Generated",
    )

    ax.set_xlabel(LABELS["strut_radius"], labelpad=8)
    ax.set_ylabel(LABELS["center_mass_radius"], labelpad=8)
    ax.set_zlabel(LABELS["corner_mass_radius"], labelpad=8)
    if bounds is not None:
        ax.set_xlim(bounds["strut_radius"])
        ax.set_ylim(bounds["center_mass_radius"])
        ax.set_zlim(bounds["corner_mass_radius"])
    ax.view_init(elev=22, azim=-52)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_distribution(
    train: pd.DataFrame,
    generated: pd.DataFrame,
    bounds: dict[str, list[float]] | None,
    path: Path,
) -> None:
    pairs = [
        ("strut_radius", "center_mass_radius"),
        ("strut_radius", "corner_mass_radius"),
        ("center_mass_radius", "corner_mass_radius"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (x_col, y_col) in zip(axes, pairs):
        ax.scatter(
            train[x_col],
            train[y_col],
            s=20,
            c="black",
            alpha=0.55,
            edgecolors="none",
            label="Training",
        )
        ax.scatter(
            generated[x_col],
            generated[y_col],
            s=26,
            c="tab:red",
            alpha=0.70,
            edgecolors="none",
            label="Generated",
        )
        ax.set_xlabel(LABELS[x_col])
        ax.set_ylabel(LABELS[y_col])
        set_pair_limits(ax, x_col, y_col, bounds)
        ax.grid(True, color="0.9", lw=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_histograms(train: pd.DataFrame, generated: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    for ax, col in zip(axes, GEOMETRY_COLS):
        ax.hist(train[col], bins=18, density=True, alpha=0.55, color="black", label="Training")
        ax.hist(generated[col], bins=18, density=True, alpha=0.45, color="tab:red", label="Generated")
        ax.set_xlabel(LABELS[col])
        ax.set_ylabel("Density")
        ax.grid(True, color="0.92", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    train_path = args.train_designs or project_dir / "outputs" / "data" / "train_designs.csv"
    generated_path = args.generated_designs or project_dir / "outputs" / "diffusion" / "generated_designs_projected.csv"
    raw_generated_path = args.raw_generated_designs or project_dir / "outputs" / "diffusion" / "generated_designs_raw.csv"
    metadata_path = args.metadata or project_dir / "outputs" / "diffusion" / "metadata.json"
    out_dir = args.out_dir or project_dir / "outputs" / "diffusion" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    generated = pd.read_csv(generated_path)
    raw_generated = pd.read_csv(raw_generated_path) if raw_generated_path.exists() else None
    generated_plot = sample_generated(generated, args.max_generated, args.seed)
    bounds = load_bounds(metadata_path)

    plot_3d_distribution(
        train,
        generated_plot,
        bounds,
        out_dir / "fig_diffusion_geometry_distribution_3d.png",
    )
    plot_pairwise_distribution(
        train,
        generated_plot,
        bounds,
        out_dir / "fig_diffusion_geometry_pairwise.png",
    )
    plot_histograms(
        train,
        generated,
        out_dir / "fig_diffusion_geometry_histograms.png",
    )
    plot_density_curves(
        train,
        generated,
        bounds,
        out_dir / "fig_diffusion_geometry_density.png",
    )
    save_summary(
        train,
        generated,
        raw_generated,
        out_dir / "diffusion_geometry_distribution_summary",
    )
    print(f"Saved diffusion distribution figures to: {out_dir}")


if __name__ == "__main__":
    main()
