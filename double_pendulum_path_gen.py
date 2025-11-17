from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Tuple, cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin


@dataclass
class PendulumParams:
    G: float = 9.8
    L1: float = 2.0
    L2: float = 2.0
    M1: float = 1.0
    M2: float = 1.0
    t_stop: float = 1000.0
    dt: float = 0.01

    @property
    def total_length(self) -> float:
        return self.L1 + self.L2


@dataclass
class RunContext:
    """Tracks metadata for the current batch/run of generated trajectories."""

    output_dir: Path | None = None
    total_energy: float | None = None
    initial_conditions: list[tuple[float, float, float, float]] = field(default_factory=list)
    lyapunov_exponent: float | None = None


RUN_DIR_PREFIX = "paths"
_RUN_CONTEXT = RunContext()


def _next_run_directory(prefix: str = RUN_DIR_PREFIX) -> Path:
    """Return the next available numbered directory (paths1, paths2, ...)."""
    idx = 1
    while True:
        candidate = Path(f"{prefix}{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def _ensure_run_directory() -> Path:
    """Create (if needed) and return the directory for the current run."""
    if _RUN_CONTEXT.output_dir is None:
        run_dir = _next_run_directory()
        run_dir.mkdir(parents=True, exist_ok=True)
        _RUN_CONTEXT.output_dir = run_dir
    return _RUN_CONTEXT.output_dir


def _register_initial_condition(condition: tuple[float, float, float, float]) -> None:
    """Track initial conditions so the README can list them."""
    if condition not in _RUN_CONTEXT.initial_conditions:
        _RUN_CONTEXT.initial_conditions.append(condition)


def _write_run_readme() -> None:
    """Write/overwrite README.txt summarizing the run metadata."""
    if _RUN_CONTEXT.output_dir is None:
        return
    readme_path = _RUN_CONTEXT.output_dir / "README.txt"
    energy = "unknown" if _RUN_CONTEXT.total_energy is None else f"{_RUN_CONTEXT.total_energy}"
    lines = [f"Initial total energy: {energy}"]
    if _RUN_CONTEXT.lyapunov_exponent is not None:
        lines.append(f"Estimated Lyapunov exponent: {_RUN_CONTEXT.lyapunov_exponent:.6f}")
    if _RUN_CONTEXT.initial_conditions:
        lines.extend(", ".join(f"{value}" for value in condition) for condition in _RUN_CONTEXT.initial_conditions)
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _start_new_run(total_energy: float | None, initial_conditions: Sequence[tuple[float, float, float, float]] | None) -> None:
    """Reset run context so the next batch writes into a fresh folder."""
    _RUN_CONTEXT.output_dir = None
    _RUN_CONTEXT.total_energy = total_energy
    _RUN_CONTEXT.lyapunov_exponent = None
    if initial_conditions:
        _RUN_CONTEXT.initial_conditions = [tuple(condition) for condition in initial_conditions]
    else:
        _RUN_CONTEXT.initial_conditions = []


def derivs(t: float, state: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Compute time derivatives for the double pendulum system."""
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (params.M1 + params.M2) * params.L1 - params.M2 * params.L1 * cos(delta) * cos(delta)
    dydx[1] = (
        params.M2 * params.L1 * state[1] * state[1] * sin(delta) * cos(delta)
        + params.M2 * params.G * sin(state[2]) * cos(delta)
        + params.M2 * params.L2 * state[3] * state[3] * sin(delta)
        - (params.M1 + params.M2) * params.G * sin(state[0])
    ) / den1

    dydx[2] = state[3]

    den2 = (params.L2 / params.L1) * den1
    dydx[3] = (
        -params.M2 * params.L2 * state[3] * state[3] * sin(delta) * cos(delta)
        + (params.M1 + params.M2) * params.G * sin(state[0]) * cos(delta)
        - (params.M1 + params.M2) * params.L1 * state[1] * state[1] * sin(delta)
        - (params.M1 + params.M2) * params.G * sin(state[2])
    ) / den2

    return dydx


def normalize_init_state(init_state: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(init_state) != 4:
        raise ValueError("Initial state must contain exactly four values: theta1, w1, theta2, w2.")

    normalized: list[float] = []
    for value in init_state:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("Initial state values must be numeric.") from exc
        if not math.isfinite(numeric_value):
            raise ValueError("Initial state values must be finite numbers.")
        normalized.append(numeric_value)

    return cast(Tuple[float, float, float, float], tuple(normalized))


def integrate_trajectory(params: PendulumParams, init_state_deg: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the equations of motion using a simple Euler method."""
    init_state_deg = normalize_init_state(init_state_deg)
    state = np.radians(np.array(init_state_deg, dtype=float))

    t = np.arange(0, params.t_stop, params.dt)
    trajectory = np.empty((len(t), 4))
    trajectory[0] = state

    for i in range(1, len(t)):
        trajectory[i] = trajectory[i - 1] + derivs(t[i - 1], trajectory[i - 1], params) * params.dt

    return t, trajectory


def estimate_lyapunov_exponent(
    params: PendulumParams,
    init_state_deg: Sequence[float],
    perturbation_deg: float = 1e-3,
) -> float:
    """
    Estimate the dominant Lyapunov exponent by comparing a trajectory with a slightly perturbed one.

    The approximation is crude but provides a qualitative sense of chaos at the requested energy level.
    """
    base_state = normalize_init_state(init_state_deg)
    perturbed_state = list(base_state)
    perturbed_state[0] += perturbation_deg

    t, base_traj = integrate_trajectory(params, base_state)
    _, pert_traj = integrate_trajectory(params, perturbed_state)

    deltas = np.linalg.norm(pert_traj - base_traj, axis=1)
    deltas = np.clip(deltas, 1e-15, None)

    total_time = t[-1] if t.size > 1 else params.dt
    lyapunov = float((np.log(deltas[-1]) - np.log(deltas[0])) / total_time)
    return lyapunov


def animate(params: PendulumParams, init_state: Sequence[float]) -> animation.FuncAnimation:
    """Animate the double pendulum for the provided parameters and initial state."""
    t, traj = integrate_trajectory(params, init_state)
    x1 = params.L1 * sin(traj[:, 0])
    y1 = -params.L1 * cos(traj[:, 0])
    x2 = params.L2 * sin(traj[:, 2]) + x1
    y2 = -params.L2 * cos(traj[:, 2]) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-params.total_length, params.total_length), ylim=(-params.total_length, params.total_length))
    ax.set_aspect("equal")
    ax.grid()

    line, = ax.plot([], [], "o-", lw=2)
    trace, = ax.plot([], [], ".-", lw=1, ms=2)
    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def _update(frame_index: int):
        thisx = [0, x1[frame_index], x2[frame_index]]
        thisy = [0, y1[frame_index], y2[frame_index]]

        line.set_data(thisx, thisy)
        trace.set_data(x2[:frame_index], y2[:frame_index])
        time_text.set_text(time_template % (frame_index * params.dt))
        return line, trace, time_text

    ani = animation.FuncAnimation(fig, _update, len(traj), interval=params.dt * 1000, blit=True)
    plt.show()
    return ani


def _safe_value_for_filename(value: float) -> str:
    safe_value = f"{value}".strip()
    if safe_value == "":
        raise ValueError("Initial state values cannot be empty.")
    for ch in ('/', '\\', ':', '*', '?', '"', "<", ">", "|"):
        safe_value = safe_value.replace(ch, "_")
    return safe_value


def _build_output_path(init_state: Iterable[float]) -> Path:
    safe_parts = [_safe_value_for_filename(v) for v in init_state]
    filename = f"{safe_parts[0]}-{safe_parts[1]}-{safe_parts[2]}-{safe_parts[3]}.csv"
    output_dir = _ensure_run_directory()
    return output_dir / filename


def generate_traj(params: PendulumParams, init_state: Sequence[float], verbose = True) -> Path:
    """Generate a trajectory CSV containing theta1/theta2 columns."""
    normalized_state = normalize_init_state(init_state)
    _, traj = integrate_trajectory(params, normalized_state)
    angles = traj[:, [0, 2]]
    angles = ((angles + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]
    output_path = _build_output_path(normalized_state)
    np.savetxt(output_path, angles, delimiter=",", header="theta1,theta2", comments="")
    _register_initial_condition(normalized_state)
    _write_run_readme()
    if verbose:
        print(f"Saved trajectory to {output_path}")
    return output_path


def generate_initial_conditions(
    E: float,
    n: int,
    params: PendulumParams,
    rng: np.random.Generator | None = None,
    initial_push: bool = True,
) -> list[tuple[float, float, float, float]]:
    """
    Generate n initial (theta1_deg, w1, theta2_deg, w2) tuples near/on the energy surface E.

    If initial_push is True:
        - Same logic as before: pick angles, then assign kinetic energy so that
          total energy equals E (up to numerical error).

    If initial_push is False:
        - Set w1 = w2 = 0 and choose (theta1, theta2) such that V(theta1, theta2) = E.
        - If E is unattainable as pure potential (w1=w2=0), raise a ValueError.
    """
    rng = rng or np.random.default_rng()
    inits: list[tuple[float, float, float, float]] = []

    m1, m2 = params.M1, params.M2
    l1, l2 = params.L1, params.L2
    g = params.G

    A = (m1 + m2) * g * l1
    B = m2 * g * l2

    # Potential range with zero velocities
    V_min = -A - B
    V_max = A + B

    if not initial_push:
        # With w1 = w2 = 0, energy must lie in [V_min, V_max]
        if not (V_min <= E <= V_max):
            raise ValueError(
                f"Requested energy E={E:.6g} is unattainable with w1=w2=0. "
                f"Allowed range is [{V_min:.6g}, {V_max:.6g}]."
            )

    attempts = 0
    max_attempts = n * 200  # a bit more generous for the no-push geometry

    while len(inits) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {n} initial conditions within {max_attempts} attempts."
            )

        if initial_push:
            # --- Original "push" logic: pick angles, then give them velocities ---
            th1 = rng.uniform(0, 2 * np.pi)
            th2 = rng.uniform(0, 2 * np.pi)

            V = -(m1 + m2) * g * l1 * math.cos(th1) - m2 * g * l2 * math.cos(th2)
            KE_avail = E - V
            if KE_avail <= 0:
                # This angle pair is too "high" in potential for the given E
                continue

            r = rng.random()
            KE1 = r * KE_avail
            KE2 = (1 - r) * KE_avail

            w1 = math.sqrt(2 * KE1 / ((m1 + m2) * l1 * l1))
            w2 = math.sqrt(2 * KE2 / (m2 * l2 * l2))

            # Randomize direction
            w1 = math.copysign(w1, rng.normal())
            w2 = math.copysign(w2, rng.normal())

        else:
            # --- No initial push: solve for angles on the level set V(theta1,theta2)=E ---
            th1 = rng.uniform(0, 2 * np.pi)

            # Solve for cos(th2) from -A cos(th1) - B cos(th2) = E
            cos_th1 = math.cos(th1)
            cos_th2 = -(E + A * cos_th1) / B

            if abs(cos_th2) > 1.0:
                # For this th1, the level set doesn't intersect; try again.
                continue

            # Two possible th2 values; choose one at random
            base = math.acos(max(-1.0, min(1.0, cos_th2)))  # clamp for safety
            if rng.random() < 0.5:
                th2 = base
            else:
                th2 = 2 * math.pi - base

            w1 = 0.0
            w2 = 0.0

        inits.append((math.degrees(th1), w1, math.degrees(th2), w2))

    _start_new_run(E, inits)
    if inits:
        try:
            lyap_estimate = estimate_lyapunov_exponent(params, inits[0])
            _RUN_CONTEXT.lyapunov_exponent = lyap_estimate
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"Warning: failed to estimate Lyapunov exponent: {exc}")
    return inits


def main() -> None:
    params = PendulumParams()
    target_energy = 12.0  # Joules
    init_conditions = generate_initial_conditions(target_energy, 1_000, params, initial_push=True)
    output_paths = [generate_traj(params, init_state) for init_state in init_conditions]
    print(f"\nSaved {len(output_paths)} trajectories.")
    # print(init_conditions)


if __name__ == "__main__":
    main()
