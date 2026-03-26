from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


OBSERVABLES = ("theta1", "w1", "theta2", "w2")
PATH_GROUPS = ("paths1", "paths2")


@dataclass(frozen=True)
class Config:
    input_root: Path = Path("gaussian_empirical_process_output2")
    output_root: Path = Path("pairwise_gaussian_estimate1")
    n_gaussian_sims: int = 5000
    random_seed: int = 123


def ensure_symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def nearest_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    symmetric = ensure_symmetric(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, eps, None)
    return (eigenvectors * clipped) @ eigenvectors.T


def load_grid_and_sigma(input_dir: Path, observable: str) -> Tuple[np.ndarray, np.ndarray]:
    grid_path = input_dir / f"{observable}_grid.npy"
    sigma_path = input_dir / f"{observable}_Sigma.npy"

    if not grid_path.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_path}")
    if not sigma_path.exists():
        raise FileNotFoundError(f"Missing Sigma file: {sigma_path}")

    grid = np.load(grid_path)
    sigma = np.load(sigma_path)
    return grid, sigma


def validate_inputs(grid: np.ndarray, sigma: np.ndarray, observable: str, group: str) -> None:
    if grid.ndim != 1:
        raise ValueError(f"{group}/{observable}: grid must be one-dimensional, got shape {grid.shape}")
    if sigma.ndim != 2:
        raise ValueError(f"{group}/{observable}: Sigma must be two-dimensional, got shape {sigma.shape}")
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"{group}/{observable}: Sigma must be square, got shape {sigma.shape}")
    if sigma.shape[0] != grid.size:
        raise ValueError(
            f"{group}/{observable}: grid length {grid.size} does not match Sigma shape {sigma.shape}"
        )
    if grid.size < 2:
        raise ValueError(f"{group}/{observable}: grid must contain at least two points")
    if np.any(~np.isfinite(grid)):
        raise ValueError(f"{group}/{observable}: grid contains non-finite values")
    if np.any(~np.isfinite(sigma)):
        raise ValueError(f"{group}/{observable}: Sigma contains non-finite values")


def sample_gaussian_vectors(
    rng: np.random.Generator,
    sigma: np.ndarray,
    n_sims: int,
    observable: str,
    group: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(sigma.shape[0], dtype=float)

    try:
        g1 = rng.multivariate_normal(mean=mean, cov=sigma, size=n_sims, check_valid="raise")
        g2 = rng.multivariate_normal(mean=mean, cov=sigma, size=n_sims, check_valid="raise")
        return g1, g2
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(
            f"[{group}] {observable}: sampling failed with the symmetrized Sigma; "
            f"retrying with nearest-PSD correction ({exc})",
            flush=True,
        )
        sigma_psd = nearest_psd(sigma)
        g1 = rng.multivariate_normal(mean=mean, cov=sigma_psd, size=n_sims, check_valid="raise")
        g2 = rng.multivariate_normal(mean=mean, cov=sigma_psd, size=n_sims, check_valid="raise")
        return g1, g2


def simulate_pairwise_gaussian_integrals(
    grid: np.ndarray,
    sigma: np.ndarray,
    n_sims: int,
    seed: int,
    observable: str,
    group: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g1, g2 = sample_gaussian_vectors(rng, sigma, n_sims, observable, group)
    return np.trapz(np.abs(g1 - g2), x=grid, axis=1)


def process_observable(
    config: Config,
    group: str,
    observable: str,
    seed: int,
) -> None:
    input_dir = config.input_root / group
    output_dir = config.output_root / group
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{group}] {observable}: loading grid and Sigma", flush=True)
    grid, sigma_loaded = load_grid_and_sigma(input_dir, observable)
    validate_inputs(grid, sigma_loaded, observable, group)

    sigma = ensure_symmetric(np.asarray(sigma_loaded, dtype=float))
    grid = np.asarray(grid, dtype=float)

    print(
        f"[{group}] {observable}: simulating {config.n_gaussian_sims} pairwise Gaussian integrals "
        f"with seed {seed}",
        flush=True,
    )
    integrals = simulate_pairwise_gaussian_integrals(
        grid=grid,
        sigma=sigma,
        n_sims=config.n_gaussian_sims,
        seed=seed,
        observable=observable,
        group=group,
    )

    output_path = output_dir / f"{observable}_pairwise_gaussian_integrals.npy"
    np.save(output_path, integrals)
    print(f"[{group}] {observable}: saved {output_path}", flush=True)


def main() -> None:
    config = Config()
    seed_counter = 0

    for group in PATH_GROUPS:
        print(f"[{group}] starting", flush=True)
        for observable in OBSERVABLES:
            seed = config.random_seed + seed_counter
            process_observable(config=config, group=group, observable=observable, seed=seed)
            seed_counter += 1
        print(f"[{group}] complete", flush=True)

    print("All pairwise Gaussian integral simulations completed.", flush=True)


if __name__ == "__main__":
    main()
