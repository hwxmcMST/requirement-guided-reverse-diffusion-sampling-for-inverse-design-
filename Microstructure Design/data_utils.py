from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SHAPE = (28, 28)
N_PIXELS = 28 * 28
N_RESPONSE = 13
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 0
DISPLACEMENTS = np.array(
    [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
    dtype=np.float32,
)


def project_dir() -> Path:
    return Path(os.environ.get("MECH_MNIST_DIR", Path(__file__).resolve().parent))


def image_path(split: Literal["train", "test"], root: str | Path | None = None) -> Path:
    root = Path(root) if root is not None else project_dir()
    return root / "MNIST_input_files" / f"mnist_img_{split}.txt"


def energy_path(split: Literal["train", "test"], root: str | Path | None = None) -> Path:
    root = Path(root) if root is not None else project_dir()
    return root / "FEA_psi_results" / f"summary_psi_{split}_all.txt"


def load_images(
    path: str | Path,
    max_samples: int | None = None,
    image_range: Literal["zero_one", "minus_one_one"] = "zero_one",
) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32, max_rows=max_samples)
    values = np.atleast_2d(values)
    if values.shape[1] != N_PIXELS:
        raise ValueError(f"Expected {N_PIXELS} pixels per image, got {values.shape[1]}.")

    images = values.reshape(-1, 1, *IMAGE_SHAPE) / 255.0
    if image_range == "minus_one_one":
        images = images * 2.0 - 1.0
    return images.astype(np.float32)


def load_energy_curves(path: str | Path, max_samples: int | None = None) -> np.ndarray:
    curves = np.loadtxt(path, dtype=np.float32, max_rows=max_samples)
    curves = np.atleast_2d(curves)
    if curves.shape[1] != N_RESPONSE:
        raise ValueError(f"Expected {N_RESPONSE} response values, got {curves.shape[1]}.")
    return curves.astype(np.float32)


@dataclass
class CurveScaler:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, curves: np.ndarray, eps: float = 1e-8) -> "CurveScaler":
        mean = curves.mean(axis=0)
        std = np.maximum(curves.std(axis=0), eps)
        return cls(mean=mean.tolist(), std=std.tolist())

    def transform(self, curves: np.ndarray) -> np.ndarray:
        return ((curves - np.asarray(self.mean)) / np.asarray(self.std)).astype(np.float32)

    def inverse_transform(self, curves_norm: np.ndarray) -> np.ndarray:
        return (curves_norm * np.asarray(self.std) + np.asarray(self.mean)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CurveScaler":
        return cls(**json.loads(Path(path).read_text()))


def load_split(
    split: Literal["train", "test"],
    root: str | Path | None = None,
    max_samples: int | None = None,
    image_range: Literal["zero_one", "minus_one_one"] = "zero_one",
) -> tuple[np.ndarray, np.ndarray]:
    images = load_images(image_path(split, root), max_samples, image_range)
    curves = load_energy_curves(energy_path(split, root), max_samples)
    if len(images) != len(curves):
        raise ValueError(f"Image/curve count mismatch: {len(images)} vs {len(curves)}.")
    return images, curves


def split_indices(
    n_samples: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_val = int(round(len(indices) * val_fraction))
    if not 0 < n_val < n_samples:
        raise ValueError("The split must contain at least one training and one validation sample.")
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def split_train_validation(
    images: np.ndarray,
    curves: np.ndarray,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx = split_indices(len(images), val_fraction, seed)
    return images[train_idx], curves[train_idx], images[val_idx], curves[val_idx]


def load_surrogate_data(
    root: str | Path | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, np.ndarray | CurveScaler]:
    x_train_all, y_train_all = load_split("train", root, max_train, "zero_one")
    x_train, y_train, x_val, y_val = split_train_validation(
        x_train_all, y_train_all, val_fraction, seed
    )
    x_test, y_test = load_split("test", root, max_test, "zero_one")
    scaler = CurveScaler.fit(y_train)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_norm": scaler.transform(y_train),
        "x_val": x_val,
        "y_val": y_val,
        "y_val_norm": scaler.transform(y_val),
        "x_test": x_test,
        "y_test": y_test,
        "y_test_norm": scaler.transform(y_test),
        "curve_scaler": scaler,
    }


def load_diffusion_data(
    root: str | Path | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, np.ndarray]:
    x_train_all = load_images(image_path("train", root), max_train, "minus_one_one")
    train_idx, val_idx = split_indices(len(x_train_all), val_fraction, seed)
    x_test = load_images(image_path("test", root), max_test, "minus_one_one")
    return {
        "x_train": x_train_all[train_idx],
        "x_val": x_train_all[val_idx],
        "x_test": x_test,
    }


def load_all_data(*args, **kwargs) -> dict[str, np.ndarray | CurveScaler]:
    return load_surrogate_data(*args, **kwargs)


class SurrogateDataset(Dataset):
    def __init__(self, images: np.ndarray, curves: np.ndarray):
        self.images = torch.from_numpy(images.astype(np.float32))
        self.curves = torch.from_numpy(curves.astype(np.float32))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.curves[idx]


class DiffusionDataset(Dataset):
    def __init__(self, images: np.ndarray):
        self.images = torch.from_numpy(images.astype(np.float32))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


def make_surrogate_dataloaders(
    root: str | Path | None = None,
    batch_size: int = 256,
    max_train: int | None = None,
    max_test: int | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, CurveScaler]:
    data = load_surrogate_data(root, max_train, max_test, val_fraction, seed)
    train_ds = SurrogateDataset(data["x_train"], data["y_train_norm"])
    val_ds = SurrogateDataset(data["x_val"], data["y_val_norm"])
    test_ds = SurrogateDataset(data["x_test"], data["y_test_norm"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, data["curve_scaler"]


def make_diffusion_dataloaders(
    root: str | Path | None = None,
    batch_size: int = 256,
    max_train: int | None = None,
    max_test: int | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = load_diffusion_data(root, max_train, max_test, val_fraction, seed)
    train_ds = DiffusionDataset(data["x_train"])
    val_ds = DiffusionDataset(data["x_val"])
    test_ds = DiffusionDataset(data["x_test"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def make_dataloaders(*args, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader, CurveScaler]:
    return make_surrogate_dataloaders(*args, **kwargs)


def save_metadata(path: str | Path, scaler: CurveScaler) -> None:
    metadata = {
        "image_shape": IMAGE_SHAPE,
        "n_response": N_RESPONSE,
        "displacements": DISPLACEMENTS.tolist(),
        "curve_scaler": asdict(scaler),
    }
    Path(path).write_text(json.dumps(metadata, indent=2))


def _plot_ready_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image).squeeze()
    if image.min() < 0.0:
        image = (image + 1.0) / 2.0
    return np.clip(image, 0.0, 1.0)


def save_image_grid(images: np.ndarray, path: str | Path, n_images: int = 16) -> None:
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

    n_images = min(n_images, len(images))
    if n_images < 1:
        raise ValueError("At least one image is required for an image grid.")
    n_cols = min(4, n_images)
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, image in zip(axes, images[:n_images]):
        ax.imshow(_plot_ready_image(image), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    for ax in axes[n_images:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_energy_curve_plot(curves: np.ndarray, path: str | Path, n_curves: int = 20) -> None:
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

    n_curves = min(n_curves, len(curves))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for curve in curves[:n_curves]:
        ax.plot(DISPLACEMENTS, curve, alpha=0.55, linewidth=1.2)
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Strain energy")
    ax.set_title(f"Mechanical MNIST energy curves, n={n_curves}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_data_visualization(
    out_dir: str | Path,
    root: str | Path | None = None,
    split: Literal["train", "test"] = "train",
    n_images: int = 16,
    n_curves: int = 20,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_samples = max(n_images, n_curves)
    images, curves = load_split(split, root, max_samples, "zero_one")

    image_grid_path = out_dir / f"{split}_image_grid.png"
    curve_plot_path = out_dir / f"{split}_energy_curves.png"
    save_image_grid(images, image_grid_path, n_images)
    save_energy_curve_plot(curves, curve_plot_path, n_curves)
    return {
        "image_grid": str(image_grid_path),
        "energy_curves": str(curve_plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/data_preview")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--n-curves", type=int, default=20)
    args = parser.parse_args()

    paths = save_data_visualization(
        args.out_dir, args.root, args.split, args.n_images, args.n_curves
    )
    print(json.dumps(paths, indent=2))


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_torch_checkpoint(path: str | Path, device="cpu") -> dict:
    """Load a trusted project checkpoint and identify incomplete model files."""
    path = Path(path)
    try:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)
    except RuntimeError as exc:
        if "PytorchStreamReader" in str(exc):
            raise RuntimeError(
                f"Cannot read checkpoint {path}: the model archive is incomplete or damaged. "
                "Restore a complete checkpoint before sampling."
            ) from exc
        raise


if __name__ == "__main__":
    main()
