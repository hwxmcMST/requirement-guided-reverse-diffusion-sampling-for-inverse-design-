from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = os.environ.get(
    "ACOUSTIC_CSV_PATH",
    "/Users/lin.ke/Desktop/project/case_1_3/band_info_2D.csv",
)

RAW_TO_CLEAN = {
    "Strut Radius": "strut_radius",
    "Center Mass Radius": "center_mass_radius",
    "Corner Mass Radius": "corner_mass_radius",
    "Mode": "mode",
    "Wavevector": "wavevector",
    "Frequency": "frequency",
}

GEOMETRY_COLS = ["strut_radius", "center_mass_radius", "corner_mass_radius"]
QUERY_COLS = GEOMETRY_COLS + ["mode", "wavevector"]
TARGET_COL = "frequency"

SQRT2 = float(np.sqrt(2.0))
STRUT_MIN = 4.0
STRUT_MAX = 6.41
CENTER_MAX = 20.0
CORNER_MAX = 20.0


def load_acoustic_data(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    train_size: int = 400,
    val_size: int = 100,
    seed: int = 7,
    check_complete: bool = True,
) -> dict[str, object]:
    """Load CSV, clean columns, assign design ids, split by design, fit normalizers."""
    df = pd.read_csv(csv_path).rename(columns=RAW_TO_CLEAN)
    missing = [col for col in RAW_TO_CLEAN.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[list(RAW_TO_CLEAN.values())].dropna().copy()
    df[QUERY_COLS + [TARGET_COL]] = df[QUERY_COLS + [TARGET_COL]].apply(pd.to_numeric)
    df["mode"] = df["mode"].astype(int)

    designs = df[GEOMETRY_COLS].drop_duplicates().sort_values(GEOMETRY_COLS).reset_index(drop=True)
    designs.insert(0, "design_id", np.arange(len(designs)))
    df = df.merge(designs, on=GEOMETRY_COLS, how="left", validate="many_to_one")
    df = df.sort_values(["design_id", "mode", "wavevector"]).reset_index(drop=True)

    modes = sorted(df["mode"].unique().astype(int).tolist())
    wavevectors = np.asarray(sorted(df["wavevector"].unique()), dtype=np.float32)
    if check_complete:
        check_band_completeness(df, modes, wavevectors, raise_error=True)

    train_ids, val_ids = split_designs(designs["design_id"].to_numpy(), train_size, val_size, seed)
    train_rows = df[df["design_id"].isin(train_ids)].copy()
    val_rows = df[df["design_id"].isin(val_ids)].copy()
    train_designs = designs[designs["design_id"].isin(train_ids)].copy()
    val_designs = designs[designs["design_id"].isin(val_ids)].copy()

    metadata = {
        "seed": seed,
        "geometry_cols": GEOMETRY_COLS,
        "query_cols": QUERY_COLS,
        "target_col": TARGET_COL,
        "modes": modes,
        "wavevectors": wavevectors.tolist(),
        "expected_rows_per_design": len(modes) * len(wavevectors),
        "train_design_ids": train_ids.tolist(),
        "val_design_ids": val_ids.tolist(),
        "geometry_bounds": geometry_bounds(designs),
        "scalers": fit_scalers(train_rows, train_designs),
    }

    return {
        "df": df,
        "designs": designs,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_designs": train_designs,
        "val_designs": val_designs,
        "metadata": metadata,
    }


def split_designs(
    design_ids: np.ndarray,
    train_size: int = 400,
    val_size: int = 100,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(design_ids).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if train_size + val_size > len(ids):
        raise ValueError(f"Need {train_size + val_size} designs, found {len(ids)}.")
    return np.sort(ids[:train_size]), np.sort(ids[train_size : train_size + val_size])


def fit_scaler(values: pd.DataFrame | np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": arr.mean(axis=0).tolist(), "std": std.tolist()}


def normalize(values: pd.DataFrame | np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return ((arr - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])).astype(np.float32)


def denormalize(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr * np.asarray(scaler["std"]) + np.asarray(scaler["mean"])).astype(np.float32)


def fit_scalers(train_rows: pd.DataFrame, train_designs: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    return {
        "geometry": fit_scaler(train_designs[GEOMETRY_COLS]),
        "query": fit_scaler(train_rows[QUERY_COLS]),
        "frequency": fit_scaler(train_rows[[TARGET_COL]]),
    }


def prepare_diffusion_data(data: dict[str, object], normalize_data: bool = True) -> dict[str, object]:
    scalers = data["metadata"]["scalers"]  # type: ignore[index]
    train = data["train_designs"][GEOMETRY_COLS]  # type: ignore[index]
    val = data["val_designs"][GEOMETRY_COLS]  # type: ignore[index]
    if normalize_data:
        train = normalize(train, scalers["geometry"])
        val = normalize(val, scalers["geometry"])
    return {
        "x_train_norm": np.asarray(train, dtype=np.float32),
        "x_val_norm": np.asarray(val, dtype=np.float32),
        "scaler": scalers["geometry"],
    }


def prepare_surrogate_data(data: dict[str, object], normalize_data: bool = True) -> dict[str, object]:
    scalers = data["metadata"]["scalers"]  # type: ignore[index]
    train_rows = data["train_rows"]  # type: ignore[assignment]
    val_rows = data["val_rows"]  # type: ignore[assignment]

    x_train, y_train = train_rows[QUERY_COLS], train_rows[[TARGET_COL]]
    x_val, y_val = val_rows[QUERY_COLS], val_rows[[TARGET_COL]]
    if normalize_data:
        x_train = normalize(x_train, scalers["query"])
        y_train = normalize(y_train, scalers["frequency"])
        x_val = normalize(x_val, scalers["query"])
        y_val = normalize(y_val, scalers["frequency"])

    return {
        "x_train_norm": np.asarray(x_train, dtype=np.float32),
        "y_train_norm": np.asarray(y_train, dtype=np.float32),
        "x_val_norm": np.asarray(x_val, dtype=np.float32),
        "y_val_norm": np.asarray(y_val, dtype=np.float32),
        "x_scaler": scalers["query"],
        "y_scaler": scalers["frequency"],
    }


def geometry_bounds(designs: pd.DataFrame | None = None) -> dict[str, list[float]]:
    return {
        "strut_radius": [STRUT_MIN, STRUT_MAX],
        "center_mass_radius": [SQRT2 * STRUT_MIN, CENTER_MAX],
        "corner_mass_radius": [(SQRT2 + 1.0) * STRUT_MIN, CORNER_MAX],
    }


def geometry_lower_bounds(strut_radius: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strut = np.asarray(strut_radius, dtype=float)
    return SQRT2 * strut, (SQRT2 + 1.0) * strut


def check_design_validity(
    designs: pd.DataFrame | np.ndarray,
    bounds: dict[str, list[float]],
) -> pd.DataFrame:
    design_df = pd.DataFrame(designs, columns=GEOMETRY_COLS) if not isinstance(designs, pd.DataFrame) else designs.copy()
    is_valid = np.ones(len(design_df), dtype=bool)
    reasons = [[] for _ in range(len(design_df))]

    strut = design_df["strut_radius"].to_numpy(dtype=float)
    center = design_df["center_mass_radius"].to_numpy(dtype=float)
    corner = design_df["corner_mass_radius"].to_numpy(dtype=float)
    center_lower, corner_lower = geometry_lower_bounds(strut)

    checks = {
        "strut_radius": ~np.isfinite(strut) | (strut < STRUT_MIN) | (strut > STRUT_MAX),
        "center_mass_radius": ~np.isfinite(center) | (center < center_lower) | (center > CENTER_MAX),
        "corner_mass_radius": ~np.isfinite(corner) | (corner < corner_lower) | (corner > CORNER_MAX),
    }
    messages = {
        "strut_radius": f"strut_radius must satisfy {STRUT_MIN} <= r <= {STRUT_MAX}",
        "center_mass_radius": "center_mass_radius must satisfy sqrt(2) * strut_radius <= r <= 20",
        "corner_mass_radius": "corner_mass_radius must satisfy (sqrt(2) + 1) * strut_radius <= r <= 20",
    }

    for col, bad in checks.items():
        is_valid &= ~bad
        for idx in np.where(bad)[0]:
            reasons[idx].append(messages[col])

    impossible_lower = (center_lower > CENTER_MAX) | (corner_lower > CORNER_MAX)
    for idx in np.where(impossible_lower)[0]:
        is_valid[idx] = False
        reasons[idx].append("strut_radius makes center/corner lower bound exceed upper bound")

    design_df["is_valid"] = is_valid
    design_df["reason"] = ["; ".join(x) for x in reasons]
    return design_df


def project_designs(
    designs: pd.DataFrame | np.ndarray,
    bounds: dict[str, list[float]],
    eps: float = 1e-6,
) -> pd.DataFrame:
    design_df = pd.DataFrame(designs, columns=GEOMETRY_COLS) if not isinstance(designs, pd.DataFrame) else designs.copy()
    design_df["strut_radius"] = design_df["strut_radius"].clip(STRUT_MIN, STRUT_MAX)
    center_lower, corner_lower = geometry_lower_bounds(design_df["strut_radius"])
    design_df["center_mass_radius"] = np.maximum(
        design_df["center_mass_radius"].clip(upper=CENTER_MAX),
        center_lower + eps,
    )
    design_df["corner_mass_radius"] = np.maximum(design_df["corner_mass_radius"].clip(upper=CORNER_MAX), corner_lower + eps)
    return design_df


def check_band_completeness(
    df: pd.DataFrame,
    modes: list[int],
    wavevectors: np.ndarray,
    raise_error: bool = False,
) -> pd.DataFrame:
    expected = len(modes) * len(wavevectors)
    n_pairs = df.groupby("design_id")[["mode", "wavevector"]].apply(lambda x: x.drop_duplicates().shape[0])
    report = (
        df.groupby("design_id")
        .agg(
            n_rows=(TARGET_COL, "size"),
            n_modes=("mode", "nunique"),
            n_wavevectors=("wavevector", "nunique"),
        )
        .reset_index()
    )
    report["n_pairs"] = report["design_id"].map(n_pairs)
    report["expected_rows"] = expected
    report["is_complete"] = (
        (report["n_rows"] == expected)
        & (report["n_pairs"] == expected)
        & (report["n_modes"] == len(modes))
        & (report["n_wavevectors"] == len(wavevectors))
    )
    if raise_error and not report["is_complete"].all():
        bad = report.loc[~report["is_complete"], "design_id"].head(10).tolist()
        raise ValueError(f"Incomplete band structure for design ids: {bad}")
    return report


def save_metadata(metadata: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_prepared_data(data: dict[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(data["metadata"], output_dir / "metadata.json")  # type: ignore[arg-type]

    data["train_designs"].to_csv(output_dir / "train_designs.csv", index=False)  # type: ignore[union-attr]
    data["val_designs"].to_csv(output_dir / "val_designs.csv", index=False)  # type: ignore[union-attr]

    diffusion = prepare_diffusion_data(data)
    surrogate = prepare_surrogate_data(data)
    np.savez_compressed(
        output_dir / "diffusion_data.npz",
        x_train_norm=diffusion["x_train_norm"],
        x_val_norm=diffusion["x_val_norm"],
    )
    np.savez_compressed(
        output_dir / "surrogate_data.npz",
        x_train_norm=surrogate["x_train_norm"],
        y_train_norm=surrogate["y_train_norm"],
        x_val_norm=surrogate["x_val_norm"],
        y_val_norm=surrogate["y_val_norm"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare acoustic band-structure data.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = load_acoustic_data(args.csv, args.train_size, args.val_size, args.seed)
    save_prepared_data(data, args.output_dir)
    print(json.dumps(data["metadata"], indent=2))


if __name__ == "__main__":
    main()
