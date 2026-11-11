from __future__ import annotations

import argparse
import sys
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


DEFAULT_PROJECT_DIR = Path("/Users/lin.ke/Desktop/project/case_1_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reconstructed band-structure curves for the acoustic surrogate."
    )
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-designs", type=int, default=1)
    parser.add_argument("--design-id", type=int, default=None)
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--subplot-height", type=float, default=5.6)
    return parser.parse_args()


def predict_numpy(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    checkpoint_path = args.checkpoint or project_dir / "outputs" / "surrogate" / "surrogate_best.pt"
    out_dir = args.out_dir or project_dir / "outputs" / "surrogate" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_dir))
    from data_utils import DEFAULT_CSV_PATH, denormalize, load_acoustic_data, prepare_surrogate_data
    from surrogate import ForwardSurrogate

    csv_path = args.csv or Path(DEFAULT_CSV_PATH)
    data = load_acoustic_data(
        csv_path=csv_path,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    prepared = prepare_surrogate_data(data, normalize_data=True)

    x_val = prepared["x_val_norm"]
    val_rows = data["val_rows"].copy()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ForwardSurrogate(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_pred_norm = predict_numpy(model, x_val, device, args.batch_size)
    y_pred = denormalize(y_pred_norm, prepared["y_scaler"])
    val_rows["frequency_pred"] = y_pred.reshape(-1)

    for stale_name in ("fig1_validation_parity.png", "fig1_validation_parity.pdf"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # Reconstructed band-structure responses for unseen validation designs.
    val_design_ids = np.asarray(data["metadata"]["val_design_ids"], dtype=int)
    if args.design_id is not None:
        if args.design_id not in val_design_ids:
            raise ValueError(f"design_id {args.design_id} is not in the validation split.")
        chosen_design_ids = np.asarray([args.design_id], dtype=int)
    else:
        chosen_design_ids = val_design_ids[: args.n_designs]
    n_rows = len(chosen_design_ids)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(args.figure_width, args.subplot_height * n_rows),
        sharex=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    for ax, design_id in zip(axes, chosen_design_ids):
        rows = val_rows[val_rows["design_id"] == design_id].sort_values(["mode", "wavevector"])
        modes = sorted(rows["mode"].unique())

        for mode in modes:
            mode_rows = rows[rows["mode"] == mode]
            wavevector = mode_rows["wavevector"].to_numpy()
            reference = mode_rows["frequency"].to_numpy()
            predicted = mode_rows["frequency_pred"].to_numpy()

            ax.plot(wavevector, reference, color="black", lw=1.15, alpha=0.88)
            ax.plot(wavevector, predicted, color="tab:red", lw=1.05, alpha=0.88, ls="--")

        geom = rows[["strut_radius", "center_mass_radius", "corner_mass_radius"]].iloc[0]
        ax.set_title(
            f"Validation design {design_id}: "
            f"$r_s$={geom['strut_radius']:.3f}, "
            f"$r_c$={geom['center_mass_radius']:.3f}, "
            f"$r_m$={geom['corner_mass_radius']:.3f}",
            fontsize=12,
            pad=8,
        )
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, color="0.9", lw=0.6)

    axes[-1].set_xlabel("Wave-vector path", fontsize=11)
    axes[-1].set_xticks([0, 1, 2, 3], [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    reference_handle = plt.Line2D([], [], color="black", lw=1.4, label="Reference")
    surrogate_handle = plt.Line2D([], [], color="tab:red", lw=1.4, ls="--", label="Surrogate")
    fig.legend(
        handles=[reference_handle, surrogate_handle],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.12,
        top=0.84 if n_rows == 1 else 0.90,
        hspace=0.42,
    )
    fig.savefig(out_dir / "fig_reconstructed_band_structure.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_reconstructed_band_structure.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
