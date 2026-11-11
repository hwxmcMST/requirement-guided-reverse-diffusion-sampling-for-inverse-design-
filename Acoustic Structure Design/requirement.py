from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_STOPBAND_THRESHOLD = 1.0
DEFAULT_FREQ_RANGE = (0.0, 60.0)


# Wavevector path in the dataset: Gamma -> X -> M -> Gamma is encoded as 0 -> 1 -> 2 -> 3.
SEGMENTS = {
    "G-X": (0.0, 1.0),
    "X-M": (1.0, 2.0),
    "M-G": (2.0, 3.0),
    "G-X-M": (0.0, 2.0),
    "X-M-G": (1.0, 3.0),
    "G-X-M-G": (0.0, 3.0),
}


# Table 2 in RAG-Inverse design-main.pdf. Frequency unit: kHz.
REQUIREMENTS = {
    "T1": {
        "stopbands": [("G-X", 45.48, 47.52)],
        "passbands": [("X-M", 37.49, 39.12)],
    },
    "T2": {
        "stopbands": [("M-G", 26.20, 27.94), ("X-M-G", 47.85, 49.91)],
        "passbands": [("M-G", 22.83, 25.49)],
    },
    "T3": {
        "stopbands": [("X-M-G", 26.57, 28.22), ("G-X", 21.91, 24.34)],
        "passbands": [("X-M", 45.27, 47.46)],
    },
    "T4": {
        "stopbands": [("G-X", 47.04, 48.39)],
        "passbands": [("G-X", 43.09, 45.96)],
    },
    "T5": {
        "stopbands": [("M-G", 45.20, 46.21)],
        "passbands": [("G-X-M-G", 28.42, 30.15)],
    },
    "T6": {
        "stopbands": [("M-G", 44.96, 47.17)],
        "passbands": [("G-X", 26.87, 28.72), ("X-M", 32.96, 34.03)],
    },
    "T7": {
        "stopbands": [("M-G", 26.95, 28.12), ("G-X-M", 32.14, 33.41)],
        "passbands": [],
    },
    "T8": {
        "stopbands": [("M-G", 27.66, 28.82), ("G-X-M", 35.54, 38.05)],
        "passbands": [],
    },
    "T9": {
        "stopbands": [("G-X-M", 37.71, 40.34), ("X-M", 24.94, 26.90)],
        "passbands": [],
    },
    "T10": {
        "stopbands": [("G-X", 30.37, 33.22), ("G-X", 46.33, 48.07)],
        "passbands": [],
    },
}


def list_requirements() -> list[str]:
    return list(REQUIREMENTS)


def get_requirement(name: str) -> dict[str, list[tuple[str, float, float]]]:
    key = name.upper()
    if key not in REQUIREMENTS:
        raise KeyError(f"Unknown requirement {name!r}. Choose from {list_requirements()}.")
    return REQUIREMENTS[key]


def response_tensor_from_dataframe(
    df: pd.DataFrame,
    id_col: str | None = None,
    value_col: str = "frequency",
) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """Return frequencies shaped as [n_designs, n_modes, n_wavevectors]."""
    id_col = resolve_id_col(df, id_col)
    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = 0

    ids = sorted(df[id_col].unique().tolist())
    modes = sorted(df["mode"].unique().astype(int).tolist())
    wavevectors = np.asarray(sorted(df["wavevector"].unique()), dtype=float)
    tensor = np.full((len(ids), len(modes), len(wavevectors)), np.nan, dtype=float)

    for i, design_id in enumerate(ids):
        table = df[df[id_col] == design_id].pivot_table(
            index="mode",
            columns="wavevector",
            values=value_col,
            aggfunc="mean",
        )
        tensor[i] = table.reindex(index=modes, columns=wavevectors).to_numpy(dtype=float)

    return tensor, ids, np.asarray(modes, dtype=int), wavevectors


def resolve_id_col(df: pd.DataFrame, id_col: str | None = None) -> str:
    if id_col is not None:
        return id_col
    for candidate in ("sample_id", "design_id"):
        if candidate in df.columns:
            return candidate
    return "response_id"


def wavevector_mask(wavevectors: np.ndarray, segment: str) -> np.ndarray:
    lo, hi = SEGMENTS[segment]
    w = wavevector_path_coordinates(wavevectors)
    return (w >= lo - 1e-8) & (w <= hi + 1e-8)


def wavevector_path_coordinates(wavevectors: np.ndarray) -> np.ndarray:
    """Accept either path coordinates 0..3 or integer indices 0..60."""
    w = np.asarray(wavevectors, dtype=float)
    if len(w) > 1 and np.nanmax(w) > 3.0 + 1e-8:
        return 3.0 * (w - np.nanmin(w)) / (np.nanmax(w) - np.nanmin(w))
    return w


def bands_calculator(
    selected_freq: np.ndarray,
    threshold: float = DEFAULT_STOPBAND_THRESHOLD,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """RAG-style passband/stopband extraction from a selected band diagram."""
    selected = np.asarray(selected_freq, dtype=float)
    if selected.ndim != 2 or selected.size == 0 or not np.isfinite(selected).all():
        return np.empty((0, 2)), np.empty((0, 2))

    passranges = np.column_stack([selected.min(axis=0), selected.max(axis=0)])
    passranges = passranges[np.argsort(passranges[:, 0])]

    stopbands = []
    first_start = passranges[0, 0]
    if first_start - freq_range[0] > threshold:
        stopbands.append([freq_range[0], first_start])

    for i in range(1, len(passranges)):
        prev_end = passranges[i - 1, 1]
        curr_start = passranges[i, 0]
        if curr_start - prev_end > threshold:
            if curr_start < freq_range[1]:
                stopbands.append([prev_end, curr_start])
            elif prev_end < freq_range[1]:
                stopbands.append([prev_end, freq_range[1]])
                break
            else:
                break

    last_end = passranges[-1, 1]
    if freq_range[1] - last_end > threshold:
        stopbands.append([last_end, freq_range[1]])

    passbands = []
    for start, end in passranges:
        if not passbands or start > passbands[-1][1]:
            passbands.append([start, end])
        else:
            passbands[-1][1] = max(passbands[-1][1], end)

    return np.asarray(stopbands, dtype=float), np.asarray(passbands, dtype=float)


def actual_bands_for_segment(
    frequencies: np.ndarray,
    wavevectors: np.ndarray,
    segment: str,
    threshold: float = DEFAULT_STOPBAND_THRESHOLD,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return actual stopbands/passbands in one target k segment."""
    freq = np.asarray(frequencies, dtype=float)
    if freq.ndim != 2:
        raise ValueError("one design must have shape [modes, wavevectors].")

    lo, hi = SEGMENTS[segment]
    w = wavevector_path_coordinates(wavevectors)
    start = int(np.argmin(np.abs(w - lo)))
    end = int(np.argmin(np.abs(w - hi)))
    start, end = sorted([start, end])
    selected = freq.T[start:end, :]
    return bands_calculator(selected, threshold, freq_range)


def condition_overlap_ratios(
    frequencies: np.ndarray,
    wavevectors: np.ndarray,
    requirement: str | dict = "T1",
    threshold: float = DEFAULT_STOPBAND_THRESHOLD,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
) -> np.ndarray:
    """Per-condition target overlap ratios, using the same band logic as RAG."""
    req = get_requirement(requirement) if isinstance(requirement, str) else requirement
    freq_batch = _as_batch(frequencies)
    conditions = [("stopband", *band) for band in req["stopbands"]]
    conditions += [("passband", *band) for band in req["passbands"]]
    ratios = np.zeros((len(freq_batch), len(conditions)), dtype=float)

    for i, freq in enumerate(freq_batch):
        for j, (band_type, segment, low, high) in enumerate(conditions):
            stopbands, passbands = actual_bands_for_segment(freq, wavevectors, segment, threshold, freq_range)
            actual = stopbands if band_type == "stopband" else passbands
            ratios[i, j] = best_interval_overlap_ratio(low, high, actual)

    return ratios


def requirement_overlap_ratio(
    frequencies: np.ndarray,
    wavevectors: np.ndarray,
    requirement: str | dict = "T1",
    threshold: float = DEFAULT_STOPBAND_THRESHOLD,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
) -> np.ndarray:
    """Minimum per-condition coverage ratio in [0, 1].

    A value of 1 means all prescribed passband/stopband conditions are
    fully covered. Using the minimum condition coverage prevents a well-matched
    condition from compensating for a poorly matched one.
    """
    ratios = condition_overlap_ratios(frequencies, wavevectors, requirement, threshold, freq_range)
    if ratios.shape[1] == 0:
        return np.zeros(ratios.shape[0], dtype=float)
    return ratios.min(axis=1)


def best_interval_overlap_ratio(low: float, high: float, intervals: np.ndarray) -> float:
    width = high - low
    return best_interval_overlap(low, high, intervals) / width if width > 0 else 0.0


def best_interval_overlap(low: float, high: float, intervals: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    overlaps = np.maximum(0.0, np.minimum(high, intervals[:, 1]) - np.maximum(low, intervals[:, 0]))
    return float(np.max(overlaps))


def hard_satisfaction_score(
    frequencies: np.ndarray,
    wavevectors: np.ndarray,
    requirement: str | dict = "T1",
    tol: float = 1e-6,
) -> np.ndarray:
    """Binary feasibility score: every condition must be fully covered."""
    ratio = requirement_overlap_ratio(frequencies, wavevectors, requirement)
    return (ratio >= 1.0 - tol).astype(float)


def evaluate_dataframe(
    df: pd.DataFrame,
    requirement: str = "T1",
    id_col: str | None = None,
    value_col: str = "frequency",
) -> pd.DataFrame:
    id_col = resolve_id_col(df, id_col)
    freq, ids, _, wavevectors = response_tensor_from_dataframe(df, id_col, value_col)
    return pd.DataFrame(
        {
            id_col: ids,
            "requirement": requirement.upper(),
            "hard_score": hard_satisfaction_score(freq, wavevectors, requirement),
        }
    )


def _as_batch(frequencies: np.ndarray) -> np.ndarray:
    freq = np.asarray(frequencies, dtype=float)
    if freq.ndim == 2:
        freq = freq[None, :, :]
    if freq.ndim != 3:
        raise ValueError("frequencies must have shape [modes, wavevectors] or [batch, modes, wavevectors].")
    return freq
