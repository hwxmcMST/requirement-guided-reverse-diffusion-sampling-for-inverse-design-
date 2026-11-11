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
    GEOMETRY_COLS,
    check_design_validity,
    denormalize,
    normalize,
    project_designs,
)
from diffusion import DiffusionSchedule, NoisePredictor, get_device, sample_normalized_designs
from requirement import (
    condition_overlap_ratios,
    get_requirement,
    list_requirements,
    requirement_overlap_ratio,
    wavevector_path_coordinates,
)
from surrogate import ForwardSurrogate, predict_numpy


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_diffusion(path: str | Path, device: torch.device, diffusion_steps: int | None = None):
    checkpoint = load_checkpoint(path, device)
    config = checkpoint["model_config"]
    model = NoisePredictor(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion_config = dict(checkpoint["diffusion_config"])
    if diffusion_steps is not None:
        diffusion_config["steps"] = diffusion_steps
    schedule = DiffusionSchedule(**diffusion_config, device=device)
    return model, schedule, checkpoint


def load_surrogate(path: str | Path, device: torch.device):
    checkpoint = load_checkpoint(path, device)
    model = ForwardSurrogate(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def decode_designs(samples_norm: np.ndarray, scaler: dict, bounds: dict):
    raw = pd.DataFrame(denormalize(samples_norm, scaler), columns=GEOMETRY_COLS)
    raw_checked = check_design_validity(raw, bounds)
    raw_checked.insert(0, "sample_id", np.arange(len(raw_checked)))

    projected = project_designs(raw, bounds)
    projected_checked = check_design_validity(projected, bounds)
    projected_checked.insert(0, "sample_id", np.arange(len(projected_checked)))
    return raw_checked, projected_checked


def predict_band_structure(
    model: torch.nn.Module,
    designs: pd.DataFrame | np.ndarray,
    modes: np.ndarray,
    wavevectors: np.ndarray,
    x_scaler: dict,
    y_scaler: dict,
    device: torch.device,
    batch_size: int,
    return_dataframe: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    design_array = np.asarray(pd.DataFrame(designs, columns=GEOMETRY_COLS)[GEOMETRY_COLS], dtype=np.float32)
    mode_grid, wave_grid = np.meshgrid(modes, wavevectors, indexing="ij")
    query_points = np.column_stack([mode_grid.reshape(-1), wave_grid.reshape(-1)]).astype(np.float32)

    x = np.repeat(design_array, len(query_points), axis=0)
    q = np.tile(query_points, (len(design_array), 1))
    x_query = np.column_stack([x, q]).astype(np.float32)

    pred_norm = predict_numpy(model, normalize(x_query, x_scaler), device, batch_size)
    pred = denormalize(pred_norm, y_scaler).reshape(len(design_array), len(modes), len(wavevectors))

    if not return_dataframe:
        return pred, pd.DataFrame()

    rows = []
    for sample_id in range(len(design_array)):
        frame = pd.DataFrame(
            {
                "sample_id": sample_id,
                "mode": mode_grid.reshape(-1),
                "wavevector": wave_grid.reshape(-1),
                "frequency": pred[sample_id].reshape(-1),
            }
        )
        rows.append(frame)
    return pred, pd.concat(rows, ignore_index=True)


def mean_pairwise_diversity(
    designs: pd.DataFrame,
    geometry_bounds: dict[str, list[float]],
    mask: np.ndarray | None = None,
) -> float:
    if mask is not None:
        designs = designs.loc[np.asarray(mask, dtype=bool)]
    if len(designs) < 2:
        return 0.0

    lower = np.asarray([geometry_bounds[col][0] for col in GEOMETRY_COLS], dtype=float)
    upper = np.asarray([geometry_bounds[col][1] for col in GEOMETRY_COLS], dtype=float)
    scale = np.maximum(upper - lower, 1e-12)
    x = (designs[GEOMETRY_COLS].to_numpy(dtype=float) - lower) / scale

    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    i_upper, j_upper = np.triu_indices(len(x), k=1)
    return float(dist[i_upper, j_upper].mean())


@torch.no_grad()
def reverse_step_candidates(
    model: torch.nn.Module,
    x: torch.Tensor,
    step: int,
    schedule: DiffusionSchedule,
    k_candidates: int,
) -> torch.Tensor:
    n, dim = x.shape
    x_flat = x[:, None, :].expand(n, k_candidates, dim).reshape(n * k_candidates, dim)
    t = torch.full((len(x_flat),), step, device=x.device, dtype=torch.long)
    beta = schedule.betas[step]
    alpha = schedule.alphas[step]
    alpha_bar = schedule.alpha_bars[step]

    pred_noise = model(x_flat, t)
    x_prev = (x_flat - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
    if step > 0:
        x_prev = x_prev + torch.sqrt(beta) * torch.randn_like(x_prev)
    return x_prev.reshape(n, k_candidates, dim)


@torch.no_grad()
def estimate_clean_x0(model: torch.nn.Module, x_noisy: torch.Tensor, step: int, schedule: DiffusionSchedule) -> torch.Tensor:
    if step <= 0:
        return x_noisy
    t = torch.full((len(x_noisy),), step, device=x_noisy.device, dtype=torch.long)
    alpha_bar = schedule.alpha_bars[step]
    pred_noise = model(x_noisy, t)
    return (x_noisy - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)


def choose_candidates(feasible: np.ndarray, condition_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    choices = np.zeros(feasible.shape[0], dtype=int)
    for i in range(feasible.shape[0]):
        feasible_idx = np.flatnonzero(feasible[i])
        if len(feasible_idx):
            choices[i] = rng.choice(feasible_idx)
            continue
        best_idx = np.flatnonzero(condition_score[i] == condition_score[i].max())
        choices[i] = rng.choice(best_idx)
    return choices


@torch.no_grad()
def guided_sample_normalized_designs(
    diffusion_model: torch.nn.Module,
    surrogate_model: torch.nn.Module,
    n_samples: int,
    k_candidates: int,
    requirement: str,
    schedule: DiffusionSchedule,
    metadata: dict,
    geometry_scaler: dict,
    x_scaler: dict,
    y_scaler: dict,
    device: torch.device,
    eval_batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    modes = np.asarray(metadata["modes"], dtype=np.float32)
    wavevectors = np.asarray(metadata["wavevectors"], dtype=np.float32)
    bounds = metadata["geometry_bounds"]
    x = torch.randn(n_samples, len(GEOMETRY_COLS), device=device)

    for step in reversed(range(schedule.steps)):
        candidates = reverse_step_candidates(diffusion_model, x, step, schedule, k_candidates)
        flat_candidates = candidates.reshape(n_samples * k_candidates, len(GEOMETRY_COLS))
        clean_step = max(step - 1, 0)
        clean_norm = estimate_clean_x0(diffusion_model, flat_candidates, clean_step, schedule)

        clean_designs = pd.DataFrame(
            denormalize(clean_norm.detach().cpu().numpy(), geometry_scaler),
            columns=GEOMETRY_COLS,
        )
        clean_designs = project_designs(clean_designs, bounds)
        pred, _ = predict_band_structure(
            surrogate_model,
            clean_designs,
            modes,
            wavevectors,
            x_scaler,
            y_scaler,
            device,
            eval_batch_size,
            return_dataframe=False,
        )

        condition_score = requirement_overlap_ratio(pred, wavevectors, requirement).reshape(n_samples, k_candidates)
        hard = condition_score >= 1.0 - 1e-6
        choice = choose_candidates(hard, condition_score, rng)
        x = candidates[torch.arange(n_samples, device=device), torch.as_tensor(choice, device=device)]

    return x.detach().cpu().numpy().astype(np.float32)


def evaluate_samples(
    name: str,
    samples_norm: np.ndarray,
    diffusion_checkpoint: dict,
    surrogate_model: torch.nn.Module,
    surrogate_checkpoint: dict,
    requirement: str,
    device: torch.device,
    eval_batch_size: int,
) -> dict[str, object]:
    metadata = diffusion_checkpoint["metadata"]
    raw, projected = decode_designs(samples_norm, diffusion_checkpoint["geometry_scaler"], metadata["geometry_bounds"])
    modes = np.asarray(metadata["modes"], dtype=np.float32)
    wavevectors = np.asarray(metadata["wavevectors"], dtype=np.float32)
    pred, pred_df = predict_band_structure(
        surrogate_model,
        projected[GEOMETRY_COLS],
        modes,
        wavevectors,
        surrogate_checkpoint["x_scaler"],
        surrogate_checkpoint["y_scaler"],
        device,
        eval_batch_size,
    )
    condition_score = requirement_overlap_ratio(pred, wavevectors, requirement)
    hard = (condition_score >= 1.0 - 1e-6).astype(float)
    scores = pd.DataFrame(
        {
            "sample_id": np.arange(len(projected)),
            "method": name,
            "requirement": requirement.upper(),
            "hard_score": hard,
            "condition_score": condition_score,
        }
    )
    metrics = {
        "n_samples": int(len(projected)),
        "n_satisfied": int(hard.sum()),
        "satisfaction_rate": float(hard.mean()),
        "diversity_dpair": mean_pairwise_diversity(
            projected,
            metadata["geometry_bounds"],
            mask=hard >= 1.0 - 1e-6,
        ),
    }
    return {
        "raw_designs": raw,
        "projected_designs": projected,
        "predictions": pred_df,
        "frequency_tensor": pred,
        "scores": scores,
        "metrics": metrics,
    }


def add_requirement_rectangles(ax, requirement: str) -> None:
    req = get_requirement(requirement)
    for segment, low, high in req["stopbands"]:
        lo, hi = _segment_range(segment)
        ax.fill_between([lo, hi], low, high, color="0.75", alpha=0.55)
    for segment, low, high in req["passbands"]:
        lo, hi = _segment_range(segment)
        ax.fill_between([lo, hi], low, high, color="#4C78A8", alpha=0.25)


def plot_satisfaction(metrics: dict, path: Path) -> None:
    methods = ["unguided", "guided"]
    rates = [metrics[m]["satisfaction_rate"] for m in methods]
    plt.figure(figsize=(5, 4))
    plt.bar(methods, rates, color=["0.55", "#4C78A8"])
    plt.ylabel("Hard satisfaction rate")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_best_responses(results: dict, requirement: str, wavevectors: np.ndarray, path: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    wavevector_x = wavevector_path_coordinates(wavevectors)
    req = get_requirement(requirement)
    n_stopbands = len(req["stopbands"])
    representative_rows = []
    for ax, method in zip(axes, ["unguided", "guided"]):
        scores = results[method]["scores"].copy()
        condition_ratios = condition_overlap_ratios(
            results[method]["frequency_tensor"],
            wavevectors,
            requirement,
        )
        if n_stopbands:
            scores["min_stopband_ratio"] = condition_ratios[:, :n_stopbands].min(axis=1)
        else:
            scores["min_stopband_ratio"] = 1.0

        if method == "guided":
            candidates = scores[scores["hard_score"] >= 1.0 - 1e-6]
            title_status = "satisfied"
            if candidates.empty:
                candidates = scores
                title_status = "best available"
            candidates = candidates.sort_values("condition_score", ascending=False)
        else:
            candidates = scores[scores["min_stopband_ratio"] < 1.0 - 1e-6]
            title_status = "stopband crossed"
            if candidates.empty:
                candidates = scores[scores["hard_score"] < 1.0 - 1e-6]
                title_status = "not satisfied"
            if candidates.empty:
                candidates = scores
                title_status = "best available"
            candidates = candidates.sort_values(
                ["min_stopband_ratio", "condition_score"],
                ascending=[True, False],
            )

        selected = candidates.iloc[0]
        sample_id = int(selected["sample_id"])
        designs = results[method]["projected_designs"]
        design_match = designs[designs["sample_id"] == sample_id]
        design = design_match.iloc[0] if not design_match.empty else designs.iloc[sample_id]
        representative_rows.append(
            {
                "method": method,
                "sample_id": sample_id,
                "selection": title_status,
                "hard_score": float(selected["hard_score"]),
                "condition_score": float(selected["condition_score"]),
                **{col: float(design[col]) for col in GEOMETRY_COLS},
            }
        )
        freq = results[method]["frequency_tensor"][sample_id]
        add_requirement_rectangles(ax, requirement)
        for mode_freq in freq:
            ax.plot(wavevector_x, mode_freq, color="black", linewidth=0.8, alpha=0.75)
        ax.set_title(f"{method}: {title_status}, hard={int(selected['hard_score'])}")
        ax.set_xlabel("Wave-vector path")
        ax.set_xticks([0, 1, 2, 3], [r"$\Gamma$", "X", "M", r"$\Gamma$"])
        ax.set_ylim(15, 55)
    axes[0].set_ylabel("Frequency (kHz)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return pd.DataFrame(representative_rows)


def save_outputs(
    results: dict,
    metrics: dict,
    out_dir: Path,
    requirement: str,
    wavevectors,
    save_debug: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    verbose_outputs = [
        "guided_designs_raw.csv",
        "guided_predicted_band_structure.csv",
        "guided_requirement_scores.csv",
        "unguided_designs_raw.csv",
        "unguided_predicted_band_structure.csv",
        "unguided_requirement_scores.csv",
    ]
    for filename in verbose_outputs:
        path = out_dir / filename
        if path.exists() and not save_debug:
            path.unlink()

    for method in ["unguided", "guided"]:
        results[method]["projected_designs"].to_csv(out_dir / f"{method}_designs_projected.csv", index=False)

    if save_debug:
        for method in ["unguided", "guided"]:
            results[method]["raw_designs"].to_csv(out_dir / f"{method}_designs_raw.csv", index=False)
            results[method]["predictions"].to_csv(out_dir / f"{method}_predicted_band_structure.csv", index=False)
            results[method]["scores"].to_csv(out_dir / f"{method}_requirement_scores.csv", index=False)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_satisfaction(metrics["methods"], out_dir / "satisfaction_comparison.png")
    representatives = plot_best_responses(results, requirement, wavevectors, out_dir / "best_predicted_band_structures.png")
    representatives.to_csv(out_dir / "representative_inverse_designs.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Requirement-guided inference for acoustic inverse design.")
    parser.add_argument("--diffusion-checkpoint", default="outputs/diffusion/diffusion_best.pt")
    parser.add_argument("--surrogate-checkpoint", default="outputs/surrogate/surrogate_best.pt")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--requirement", default="T1", choices=list_requirements())
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--k-candidates", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate CSV files for debugging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    diffusion_model, schedule, diffusion_checkpoint = load_diffusion(args.diffusion_checkpoint, device, args.diffusion_steps)
    surrogate_model, surrogate_checkpoint = load_surrogate(args.surrogate_checkpoint, device)

    unguided_norm = sample_normalized_designs(
        diffusion_model,
        args.n_samples,
        schedule,
        device,
        len(GEOMETRY_COLS),
    )
    guided_norm = guided_sample_normalized_designs(
        diffusion_model,
        surrogate_model,
        args.n_samples,
        args.k_candidates,
        args.requirement,
        schedule,
        diffusion_checkpoint["metadata"],
        diffusion_checkpoint["geometry_scaler"],
        surrogate_checkpoint["x_scaler"],
        surrogate_checkpoint["y_scaler"],
        device,
        args.eval_batch_size,
        args.seed,
    )

    results = {
        "unguided": evaluate_samples(
            "unguided",
            unguided_norm,
            diffusion_checkpoint,
            surrogate_model,
            surrogate_checkpoint,
            args.requirement,
            device,
            args.eval_batch_size,
        ),
        "guided": evaluate_samples(
            "guided",
            guided_norm,
            diffusion_checkpoint,
            surrogate_model,
            surrogate_checkpoint,
            args.requirement,
            device,
            args.eval_batch_size,
        ),
    }

    metrics = {
        "requirement": args.requirement,
        "n_samples": args.n_samples,
        "k_candidates": args.k_candidates,
        "diffusion_steps": schedule.steps,
        "methods": {
            "unguided": results["unguided"]["metrics"],
            "guided": results["guided"]["metrics"],
        },
        "note": "condition_score is the minimum per-condition coverage ratio; hard satisfaction requires condition_score = 1.",
    }

    out_dir = Path(args.output_dir)
    wavevectors = np.asarray(diffusion_checkpoint["metadata"]["wavevectors"], dtype=float)
    save_outputs(results, metrics, out_dir, args.requirement, wavevectors, args.save_debug)
    print(json.dumps(metrics, indent=2))


def _segment_range(segment: str) -> tuple[float, float]:
    ranges = {
        "G-X": (0.0, 1.0),
        "X-M": (1.0, 2.0),
        "M-G": (2.0, 3.0),
        "G-X-M": (0.0, 2.0),
        "X-M-G": (1.0, 3.0),
        "G-X-M-G": (0.0, 3.0),
    }
    return ranges[segment]


if __name__ == "__main__":
    main()
