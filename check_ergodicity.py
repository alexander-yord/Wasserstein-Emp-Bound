#!/usr/bin/env python3
"""
Utilities for checking ergodicity of observables across pendulum trajectories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".npy", ".npz"}
IGNORED_FILENAMES = {"readme.txt"}
PI2 = 2 * np.pi


def _wrap_to_pi(arr: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi). Keeps statistics bounded for periodic data."""
    return (arr + np.pi) % PI2 - np.pi


def _ensure_time_series(array: np.ndarray) -> np.ndarray:
    """Flatten to 1D floating array."""
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def _extract_angles_from_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Try to pull theta1/theta2 from different array layouts."""
    names = getattr(arr.dtype, "names", None)
    if names and {"theta1", "theta2"} <= set(names):
        return np.asarray(arr["theta1"], dtype=float), np.asarray(arr["theta2"], dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] == 2:
            return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float)
        if arr.shape[1] == 2:
            return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)
    raise ValueError("Unsupported array layout for theta1/theta2 data.")


def _load_angles(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load theta1 and theta2 arrays from supported trajectory files."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path}")

    if suffix in {".csv", ".txt"}:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        has_header = "theta" in first_line.lower()
        delimiter = "," if "," in first_line else None
        skiprows = 1 if has_header else 0
        try:
            data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, dtype=float)
        except ValueError as exc:
            raise ValueError(f"Failed to parse numeric data: {exc}") from exc
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("Expected at least two columns for theta1/theta2.")
        return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)

    if suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray):
            try:
                return _extract_angles_from_array(loaded)
            except ValueError:
                pass
            try:
                maybe_dict = loaded.item()
            except ValueError:
                maybe_dict = None
            if isinstance(maybe_dict, dict):
                return np.asarray(maybe_dict["theta1"], dtype=float), np.asarray(maybe_dict["theta2"], dtype=float)
        raise ValueError(f"Unable to parse theta arrays from {path}")

    # suffix == ".npz"
    with np.load(path, allow_pickle=True) as data:
        if {"theta1", "theta2"} <= set(data.files):
            return np.asarray(data["theta1"], dtype=float), np.asarray(data["theta2"], dtype=float)
        raise ValueError(f"theta1/theta2 not found in {path}")


def compute_ergodicity_stats(folder: str, fnc: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """
    Load trajectories and compute per-trajectory and ensemble averages for the given observable.
    Returns a dictionary with the computed statistics.
    """
    folder_path = Path(folder)
    files = sorted(
        [
            p
            for p in folder_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.name.lower() not in IGNORED_FILENAMES
        ]
    )
    if not files:
        raise FileNotFoundError(f"No supported trajectory files found in {folder_path}")

    observable_name = getattr(fnc, "__name__", repr(fnc))
    trajectories: List[str] = []
    time_averages: List[float] = []
    observables: List[np.ndarray] = []
    skipped = 0

    for path in files:
        try:
            theta1, theta2 = _load_angles(path)
        except ValueError as exc:
            print(f"[warning] Skipping {path.name}: {exc}")
            skipped += 1
            continue

        # Use wrapped angles so statistics stay within [-pi, pi) for angular observables.
        theta1 = _wrap_to_pi(theta1)
        theta2 = _wrap_to_pi(theta2)
        obs_values = _ensure_time_series(fnc(theta1, theta2))
        if obs_values.size == 0:
            raise ValueError(f"Observable evaluation returned an empty array for {path.name}")
        time_avg = float(obs_values.mean())

        trajectories.append(path.name)
        time_averages.append(time_avg)
        observables.append(obs_values)

    if not observables:
        raise RuntimeError(f"No usable trajectories found in {folder_path}")

    min_length = min(obs.size for obs in observables)
    if min_length == 0:
        raise ValueError("At least one trajectory has zero length.")
    stacked = np.vstack([obs[:min_length] for obs in observables])
    ensemble_avg = float(stacked.mean(axis=0).mean())

    time_avg_arr = np.asarray(time_averages, dtype=float)

    return {
        "folder": folder_path,
        "observable_name": observable_name,
        "trajectories": trajectories,
        "time_averages": time_avg_arr,
        "ensemble_average": ensemble_avg,
        "deviations": time_avg_arr - ensemble_avg,
        "samples_per_trajectory": min_length,
        "skipped_files": skipped,
        "num_trajectories": len(trajectories),
    }


def check_ergodicity(folder: str, fnc: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
    """
    folder: path containing many trajectory files (e.g. './trajectories')
    fnc: a lambda function taking (theta1, theta2) and returning a 1D array of observable values
    """
    try:
        stats = compute_ergodicity_stats(folder, fnc)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return

    folder_path = stats["folder"]
    observable_name = stats["observable_name"]
    trajectories = stats["trajectories"]
    time_averages = stats["time_averages"]
    deviations = stats["deviations"]
    min_length = stats["samples_per_trajectory"]
    skipped = stats["skipped_files"]

    print("=" * 72)
    print(f"Folder: {folder_path}")
    print(f"Observable: {observable_name}")
    print(f"Trajectories: {len(trajectories)} | Samples per trajectory (ensemble calc): {min_length}")
    print("-" * 72)
    print(f"{'Trajectory':40s} {'Time avg':>14s} {'Deviation':>14s}")
    for name, t_avg, deviation in zip(trajectories, time_averages, deviations):
        print(f"{name:40s} {t_avg:14.6f} {deviation:14.6f}")
    print("-" * 72)
    print(f"Ensemble average: {stats['ensemble_average']:.6f}")
    print("=" * 72)
    if skipped:
        print(f"Skipped {skipped} file(s) due to parsing issues.")
    print()


def main() -> None:
    folder = "./paths1"

    f1 = lambda th1, th2: np.cos(th1)
    f2 = lambda th1, th2: np.cos(th2)
    f3 = lambda th1, th2: np.sin(th1 - th2)
    f4 = lambda th1, th2: ((th1 % (2 * np.pi)) < (np.pi / 2)).astype(float)

    check_ergodicity(folder, f1)
    check_ergodicity(folder, f2)
    check_ergodicity(folder, f3)
    check_ergodicity(folder, f4)


if __name__ == "__main__":
    main()
