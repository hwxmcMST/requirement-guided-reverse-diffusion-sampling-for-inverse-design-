from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data_utils import DISPLACEMENTS, energy_path, load_energy_curves


DEFAULT_R1_Q_MIN = 0.6
DEFAULT_R1_Q_MAX = 0.8
DEFAULT_R2_Q = 0.3


@dataclass(frozen=True)
class Requirement:
    name: str
    kind: str
    psi_min: float | None = None
    psi_max: float | None = None
    early_u_max: float | None = None
    c1: float | None = None

    def __post_init__(self):
        if self.kind == "final_energy_range":
            if self.psi_min is None or self.psi_max is None or not np.isfinite([self.psi_min, self.psi_max]).all() or self.psi_min >= self.psi_max:
                raise ValueError("R1 requires finite psi_min < psi_max.")
        elif self.kind == "low_early_energy":
            if self.c1 is None or not np.isfinite(self.c1) or self.early_u_max is None or not np.isfinite(self.early_u_max) or not (DISPLACEMENTS <= self.early_u_max).any():
                raise ValueError("R2 requires a finite threshold and a nonempty early-loading region.")
        else:
            raise ValueError(f"Unknown requirement kind: {self.kind}")


def r1_final_energy_range(psi_min: float, psi_max: float) -> Requirement:
    return Requirement("R1", "final_energy_range", psi_min=psi_min, psi_max=psi_max)


def r2_low_early_energy(c1: float, early_u_max: float = 1.0) -> Requirement:
    return Requirement("R2", "low_early_energy", early_u_max=early_u_max, c1=c1)


def ensure_2d(curves: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves[None, :]
    if curves.ndim != 2 or curves.shape[1] != len(DISPLACEMENTS) or len(curves) == 0:
        raise ValueError(f"Expected a nonempty array of shape [N, {len(DISPLACEMENTS)}], got {curves.shape}.")
    if not np.isfinite(curves).all():
        raise ValueError("Response curves must contain only finite values.")
    return curves


def final_energy(curves: np.ndarray) -> np.ndarray:
    return ensure_2d(curves)[:, -1]


def early_max_energy(curves: np.ndarray, early_u_max: float = 1.0) -> np.ndarray:
    curves = ensure_2d(curves)
    mask = DISPLACEMENTS <= early_u_max
    if not np.isfinite(early_u_max) or not mask.any():
        raise ValueError("early_u_max must select at least one displacement.")
    return curves[:, mask].max(axis=1)


def r1_from_quantiles(
    reference_curves: np.ndarray,
    q_min: float = DEFAULT_R1_Q_MIN,
    q_max: float = DEFAULT_R1_Q_MAX,
) -> Requirement:
    values = final_energy(reference_curves)
    psi_min, psi_max = np.quantile(values, [q_min, q_max])
    return r1_final_energy_range(float(psi_min), float(psi_max))


def r2_from_quantile(
    reference_curves: np.ndarray,
    q: float = DEFAULT_R2_Q,
    early_u_max: float = 1.0,
) -> Requirement:
    c1 = np.quantile(early_max_energy(reference_curves, early_u_max), q)
    return r2_low_early_energy(float(c1), early_u_max)


def hard_satisfaction(curves: np.ndarray, requirement: Requirement) -> np.ndarray:
    curves = ensure_2d(curves)

    if requirement.kind == "final_energy_range":
        final_energy = curves[:, -1]
        return (
            (final_energy >= requirement.psi_min)
            & (final_energy <= requirement.psi_max)
        ).astype(np.int64)

    if requirement.kind == "low_early_energy":
        return (early_max_energy(curves, requirement.early_u_max) <= requirement.c1).astype(np.int64)

    raise ValueError(f"Unknown requirement kind: {requirement.kind}")


def feasible_rate(curves: np.ndarray, requirement: Requirement) -> float:
    return float(hard_satisfaction(curves, requirement).mean())


def evaluate_curves(
    curves: np.ndarray,
    requirement: Requirement,
    include_details: bool = False,
) -> dict[str, object]:
    curves = ensure_2d(curves)
    hard = hard_satisfaction(curves, requirement)
    result: dict[str, object] = {
        "requirement": asdict(requirement),
        "n_samples": int(len(curves)),
        "n_feasible": int(hard.sum()),
        "feasible_rate": float(hard.mean()),
    }
    if not include_details:
        return result

    if requirement.kind == "final_energy_range":
        result["hard_score"] = hard.tolist()
        result["final_energy"] = final_energy(curves).tolist()
    elif requirement.kind == "low_early_energy":
        result["hard_score"] = hard.tolist()
        result["early_max_energy"] = early_max_energy(curves, requirement.early_u_max).tolist()
    return result


def load_curve_file(path: str | Path) -> np.ndarray:
    path = str(path)
    return np.load(path) if path.endswith(".npy") else np.loadtxt(path, dtype=np.float32)


def parse_requirement(args: argparse.Namespace, reference_curves: np.ndarray) -> Requirement:
    if args.requirement == "R1":
        if args.psi_min is None and args.psi_max is None:
            return r1_from_quantiles(reference_curves, args.r1_q_min, args.r1_q_max)
        if args.psi_min is None or args.psi_max is None:
            raise ValueError("R1 needs both --psi-min and --psi-max, or neither.")
        return r1_final_energy_range(args.psi_min, args.psi_max)

    if args.requirement == "R2":
        if args.c1 is None:
            return r2_from_quantile(reference_curves, args.r2_q, args.early_u_max)
        return r2_low_early_energy(args.c1, args.early_u_max)

    raise ValueError(f"Unknown requirement: {args.requirement}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=str, required=True)
    parser.add_argument("--reference-curves", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--requirement", choices=["R1", "R2"], required=True)
    parser.add_argument("--psi-min", type=float, default=None)
    parser.add_argument("--psi-max", type=float, default=None)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--early-u-max", type=float, default=1.0)
    parser.add_argument("--r1-q-min", type=float, default=DEFAULT_R1_Q_MIN)
    parser.add_argument("--r1-q-max", type=float, default=DEFAULT_R1_Q_MAX)
    parser.add_argument("--r2-q", type=float, default=DEFAULT_R2_Q)
    parser.add_argument("--include-details", action="store_true")
    args = parser.parse_args()

    curves = load_curve_file(args.curves)
    reference_curves = (
        load_curve_file(args.reference_curves)
        if args.reference_curves
        else load_energy_curves(energy_path("train"))
    )
    result = evaluate_curves(
        curves, parse_requirement(args, reference_curves), args.include_details
    )

    text = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
