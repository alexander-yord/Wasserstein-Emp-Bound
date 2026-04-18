from __future__ import annotations

import os
import glob
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


OBSERVABLES = ["theta1", "w1", "theta2", "w2"]
RESULT_COLS = ["W1_1", "W1_2", "W1_3", "W1_4"]


@dataclass
class Config:
    burn_in: int = 0
    grid_size: int = 80
    lag_cutoff: int = 50
    n_gaussian_sims: int = 5000
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    use_flat_hat: bool = True
    flat_hat_fraction: float = 0.5
    random_seed: int = 123
    figure_dpi: int = 180
    bins: int = 40
    n_jobs: int = -1
    joblib_prefer: str = "threads"


# -----------------------------
# I/O helpers
# -----------------------------
def find_csvs(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder!r}")
    return files


def load_single_path_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if df.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {file_path}, got {df.shape[1]}")

    cols = list(df.columns[:4])
    rename_map = {cols[0]: "theta1", cols[1]: "w1", cols[2]: "theta2", cols[3]: "w2"}
    df = df.rename(columns=rename_map)
    return df[["theta1", "w1", "theta2", "w2"]].copy()


def load_ensemble(folder: str, burn_in: int = 0) -> Dict[str, List[np.ndarray]]:
    files = find_csvs(folder)
    out = {obs: [] for obs in OBSERVABLES}

    for fp in files:
        df = load_single_path_csv(fp)
        if burn_in > 0:
            df = df.iloc[burn_in:].reset_index(drop=True)
        for obs in OBSERVABLES:
            out[obs].append(df[obs].to_numpy(dtype=float))

    return out


def load_wasserstein_results(results_csv: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    needed = ["file_name", *RESULT_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {results_csv}: {missing}")
    return df[needed].copy()


# -----------------------------
# Empirical-process covariance estimation
# -----------------------------
def pooled_quantile_grid(paths: List[np.ndarray], grid_size: int, q_low: float, q_high: float) -> np.ndarray:
    pooled = np.concatenate(paths)
    probs = np.linspace(q_low, q_high, grid_size)
    grid = np.quantile(pooled, probs)
    grid = np.unique(grid)
    if grid.size < 5:
        raise ValueError("Grid collapsed to too few unique points; increase variability or grid design.")
    return grid.astype(float)


def empirical_cdf_on_grid(paths: List[np.ndarray], grid: np.ndarray) -> np.ndarray:
    pooled = np.sort(np.concatenate(paths))
    counts = np.searchsorted(pooled, grid, side="right")
    return counts / pooled.size


def centered_indicator_matrix(x: np.ndarray, grid: np.ndarray, Fhat: np.ndarray) -> np.ndarray:
    return (x[:, None] <= grid[None, :]).astype(float) - Fhat[None, :]


def flat_hat_weight(k: int, L: int, flat_fraction: float = 0.5) -> float:
    """
    Trapezoidal flat-top ("flat-hat") lag window.

    With x = |k| / (L + 1), the weight is
        1,                  0 <= x <= flat_fraction
        (1 - x)/(1-flat_fraction),  flat_fraction < x <= 1
        0,                  x > 1

    The default flat_fraction=0.5 gives the common flat-top shape
    w(x)=1 for x<=1/2 and w(x)=2(1-x) for 1/2 < x <= 1.
    """
    if L <= 0:
        return 1.0 if k == 0 else 0.0

    if not (0.0 < flat_fraction < 1.0):
        raise ValueError("flat_fraction must lie strictly between 0 and 1.")

    x = abs(k) / (L + 1.0)
    if x <= flat_fraction:
        return 1.0
    if x <= 1.0:
        return (1.0 - x) / (1.0 - flat_fraction)
    return 0.0


def _path_long_run_covariance(
    x: np.ndarray,
    grid: np.ndarray,
    Fhat: np.ndarray,
    L: int,
    flat_fraction: float = 0.5,
) -> np.ndarray:
    Y = centered_indicator_matrix(x, grid, Fhat)
    n = Y.shape[0]
    if n <= L + 1:
        raise ValueError(f"Path length {n} is too short for lag cutoff {L}.")

    Sigma_i = (Y.T @ Y) / n

    for k in range(1, L + 1):
        wk = flat_hat_weight(k, L, flat_fraction=flat_fraction)
        Gk = (Y[:-k].T @ Y[k:]) / (n - k)
        Sigma_i += wk * (Gk + Gk.T)

    return Sigma_i


def estimate_long_run_covariance(
    paths: List[np.ndarray],
    grid: np.ndarray,
    Fhat: np.ndarray,
    L: int,
    flat_fraction: float = 0.5,
    n_jobs: int = -1,
    prefer: str = "threads",
) -> np.ndarray:
    J = grid.size
    M = len(paths)

    if M == 0:
        return np.zeros((J, J), dtype=float)

    completed = 0
    lock = threading.Lock()

    def compute_and_report(x: np.ndarray) -> np.ndarray:
        nonlocal completed
        sigma_i = _path_long_run_covariance(x, grid, Fhat, L, flat_fraction=flat_fraction)
        with lock:
            completed += 1
            print(f"Long-run covariance progress: {completed}/{M}", flush=True)
        return sigma_i

    partial_sigmas = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(compute_and_report)(x)
        for x in paths
    )

    Sigma = np.sum(partial_sigmas, axis=0) / M
    return Sigma


def nearest_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


# -----------------------------
# Gaussian simulation and integral functional
# -----------------------------
def simulate_gaussian_integrals(grid: np.ndarray, Sigma: np.ndarray, n_sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.zeros(grid.size, dtype=float)
    Z = rng.multivariate_normal(mean=mean, cov=Sigma, size=n_sims, check_valid="warn")

    vals = np.trapz(np.abs(Z), x=grid, axis=1)
    return vals


# -----------------------------
# Diagnostics / scaling
# -----------------------------
def effective_q_from_paths(paths: List[np.ndarray]) -> int:
    return int(min(len(x) for x in paths))


def scaled_empirical_wasserstein(results_df: pd.DataFrame, obs_idx: int, q: int) -> np.ndarray:
    w = results_df[RESULT_COLS[obs_idx]].to_numpy(dtype=float)
    return np.sqrt(q) * w


def summarize_distribution(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_system_comparison(
    system_name: str,
    empirical_scaled: Dict[str, np.ndarray],
    predicted: Dict[str, np.ndarray],
    out_png: str,
    bins: int = 40,
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for j, obs in enumerate(OBSERVABLES):
        ax = axes[j]
        emp = empirical_scaled[obs]
        pred = predicted[obs]

        lo = min(emp.min(), pred.min())
        hi = max(emp.max(), pred.max())
        edges = np.linspace(lo, hi, bins + 1)

        ax.hist(emp, bins=edges, density=True, alpha=0.45, label=r"Empirical $\sqrt{q}W$")
        ax.hist(pred, bins=edges, density=True, alpha=0.45, label=r"Predicted $\int |G(t)|dt$")
        ax.set_title(obs)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(frameon=False)

    fig.suptitle(system_name)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# End-to-end run
# -----------------------------
def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def analyze_system(
    paths_folder: str,
    results_csv: str,
    out_dir: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    system_name = os.path.basename(os.path.normpath(paths_folder))
    system_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{system_name}] loading ensemble from {paths_folder}")
    ensemble = load_ensemble(paths_folder, burn_in=cfg.burn_in)
    n_paths = len(ensemble[OBSERVABLES[0]])
    print(f"[{system_name}] loaded {n_paths} paths")

    print(f"[{system_name}] loading Wasserstein results from {results_csv}")
    results_df = load_wasserstein_results(results_csv)

    predicted: Dict[str, np.ndarray] = {}
    empirical_scaled: Dict[str, np.ndarray] = {}
    rows = []

    for j, obs in enumerate(OBSERVABLES):
        obs_start = time.perf_counter()
        paths = ensemble[obs]
        print(f"[{system_name}] ({j + 1}/{len(OBSERVABLES)}) {obs}: building grid")
        grid = pooled_quantile_grid(paths, cfg.grid_size, cfg.quantile_low, cfg.quantile_high)
        Fhat = empirical_cdf_on_grid(paths, grid)
        print(
            f"[{system_name}] ({j + 1}/{len(OBSERVABLES)}) {obs}: "
            f"estimating long-run covariance with {len(paths)} paths, L={cfg.lag_cutoff}, n_jobs={cfg.n_jobs}"
        )
        Sigma = estimate_long_run_covariance(
            paths,
            grid,
            Fhat,
            cfg.lag_cutoff,
            flat_fraction=cfg.flat_hat_fraction,
            n_jobs=cfg.n_jobs,
            prefer=cfg.joblib_prefer,
        )
        print(f"[{system_name}] ({j + 1}/{len(OBSERVABLES)}) {obs}: simulating Gaussian integrals")
        Sigma_psd = nearest_psd(Sigma)
        sims = simulate_gaussian_integrals(grid, Sigma_psd, cfg.n_gaussian_sims, cfg.random_seed + j)

        q = effective_q_from_paths(paths)
        emp_scaled = scaled_empirical_wasserstein(results_df, j, q)

        predicted[obs] = sims
        empirical_scaled[obs] = emp_scaled

        np.save(os.path.join(out_dir, f"{obs}_grid.npy"), grid)
        np.save(os.path.join(out_dir, f"{obs}_Fhat.npy"), Fhat)
        np.save(os.path.join(out_dir, f"{obs}_Sigma.npy"), Sigma_psd)
        np.save(os.path.join(out_dir, f"{obs}_predicted_integrals.npy"), sims)
        np.save(os.path.join(out_dir, f"{obs}_empirical_scaled_w.npy"), emp_scaled)
        np.savez(
            os.path.join(out_dir, f"{obs}_distributions.npz"),
            predicted_integrals=sims,
            empirical_scaled_w=emp_scaled,
            grid=grid,
            Fhat=Fhat,
        )

        row = {
            "observable": obs,
            "q_used": q,
            "grid_size": int(grid.size),
            "lag_cutoff": cfg.lag_cutoff,
            "flat_hat_fraction": cfg.flat_hat_fraction,
        }
        row.update({f"emp_{k}": v for k, v in summarize_distribution(emp_scaled).items()})
        row.update({f"pred_{k}": v for k, v in summarize_distribution(sims).items()})
        rows.append(row)
        print(
            f"[{system_name}] ({j + 1}/{len(OBSERVABLES)}) {obs}: done in "
            f"{format_seconds(time.perf_counter() - obs_start)}"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    np.savez(
        os.path.join(out_dir, "all_distributions.npz"),
        **{f"{obs}_predicted_integrals": predicted[obs] for obs in OBSERVABLES},
        **{f"{obs}_empirical_scaled_w": empirical_scaled[obs] for obs in OBSERVABLES},
    )

    print(f"[{system_name}] writing comparison plot")
    plot_system_comparison(
        system_name=system_name,
        empirical_scaled=empirical_scaled,
        predicted=predicted,
        out_png=os.path.join(out_dir, f"{system_name}_comparison.png"),
        bins=cfg.bins,
        dpi=cfg.figure_dpi,
    )
    print(f"[{system_name}] complete in {format_seconds(time.perf_counter() - system_start)}")

    return summary, empirical_scaled, predicted


def main() -> None:
    cfg = Config(
        burn_in=50_000,
        grid_size=80,
        lag_cutoff=250,
        n_gaussian_sims=5000,
        quantile_low=0.01,
        quantile_high=0.99,
        random_seed=123,
        figure_dpi=180,
        bins=40,
        n_jobs=-1,
        joblib_prefer="threads",
        flat_hat_fraction=0.5,
    )

    jobs = [
        {
            "paths_folder": "paths1",
            "results_csv": os.path.join("AVMs6", "paths1_results.csv"),
            "out_dir": os.path.join("gaussian_empirical_process_output6", "paths1"),
        },
        {
            "paths_folder": "paths2",
            "results_csv": os.path.join("AVMs6", "paths2_results.csv"),
            "out_dir": os.path.join("gaussian_empirical_process_output6", "paths2"),
        }
    ]

    all_summaries = []
    run_start = time.perf_counter()
    print(f"Program beginning with {len(jobs)} ensemble(s)")
    for job in jobs:
        summary, _, _ = analyze_system(
            paths_folder=job["paths_folder"],
            results_csv=job["results_csv"],
            out_dir=job["out_dir"],
            cfg=cfg,
        )
        summary.insert(0, "system", os.path.basename(job["paths_folder"]))
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    os.makedirs("gaussian_empirical_process_output_flat_hat", exist_ok=True)
    combined.to_csv(os.path.join("gaussian_empirical_process_output_flat_hat", "combined_summary.csv"), index=False)
    print(f"Program finished in {format_seconds(time.perf_counter() - run_start)}")
    print(combined)


if __name__ == "__main__":
    main()
