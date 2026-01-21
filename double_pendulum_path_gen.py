from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Tuple, cast
from scipy.integrate import solve_ivp

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


def integrate_trajectory_euler(params: PendulumParams, init_state: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the equations of motion using a simple Euler method."""

    t = np.arange(0, params.t_stop, params.dt)
    trajectory = np.empty((len(t), 4))
    trajectory[0] = init_state

    for i in range(1, len(t)):
        trajectory[i] = trajectory[i - 1] + derivs(t[i - 1], trajectory[i - 1], params) * params.dt

    return t, trajectory

def integrate_trajectory(params: PendulumParams, init_state: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Functions that integrates the trajectory using Runge Kutta """

    def f(t, y):
        return derivs(t, y, params)

    sol = solve_ivp(
        f,
        t_span=(0.0, params.t_stop),
        y0=init_state,
        method="DOP853",   # or "RK45"
        max_step=params.dt # controls resolution of saved points
    )

    t = sol.t
    traj = sol.y.T  # shape (n_points, 4)
    return t, traj

def estimate_lyapunov_exponent(
    params: PendulumParams,
    init_state: Sequence[float],
    perturbation_deg: float = 1e-3,
) -> float:
    """
    Estimate the dominant Lyapunov exponent by comparing a trajectory with a slightly perturbed one.

    The approximation is crude but provides a qualitative sense of chaos at the requested energy level.
    """
    base_state = normalize_init_state(init_state)
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
    """Generate a trajectory CSV containing theta1, w1, theta2, w2 columns."""
    normalized_state = normalize_init_state(init_state)
    t, traj = integrate_trajectory(params, normalized_state)
    
    # 1. Create a copy of the full trajectory (all 4 columns)
    #    Assumes order: [theta1, w1, theta2, w2]
    output_data = traj.copy()

    # 2. Wrap ONLY the angles (columns 0 and 2) to [-pi, pi]
    #    Do NOT wrap the velocities (columns 1 and 3)
    output_data[:, 0] = ((output_data[:, 0] + np.pi) % (2 * np.pi)) - np.pi
    output_data[:, 2] = ((output_data[:, 2] + np.pi) % (2 * np.pi)) - np.pi

    output_path = _build_output_path(normalized_state)
    
    # 3. Save full array with updated header
    np.savetxt(output_path, output_data, delimiter=",", header="theta1,w1,theta2,w2", comments="")
    
    _register_initial_condition(normalized_state)
    _write_run_readme()
    
    if verbose:
        print(f"Saved trajectory to {output_path}. Final time: t={t[-1]}")
    return output_path


def generate_initial_conditions(
    E: float,
    n: int,
    params: PendulumParams
) -> list[tuple[float, float, float, float]]:
    """Function to generate n sets of initial conditions (th1, w1, th2, w2) all satisfying the same initial energy E"""
    inits: list[tuple[float, float, float, float]] = []

    # 1. Input parameters
    m1, m2 = params.M1, params.M2
    l1, l2 = params.L1, params.L2
    g = params.G
    M = m1 + m2

    while len(inits) < n:
        # 2. Random Angles
        th1 = np.random.uniform(-np.pi, np.pi)
        th2 = np.random.uniform(-np.pi, np.pi)

        # 3. Check Potential Energy Viability
        V = -M * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        
        # If Potential Energy is greater than Total Energy, kinetic energy would be negative (impossible)
        if V > E:
            continue

        # 4. Determine Kinetic Energy Bounds (w1_max)
        # Using the discriminant condition B^2 - AC >= 0
        delta = th1 - th2
        term_denominator = l1**2 * (m1 + m2 * (np.sin(delta)**2))
        
        # Ensure we don't divide by zero (though physically unlikely with mass > 0)
        if term_denominator <= 0:
            continue
            
        w1_max = np.sqrt((2 * (E - V)) / term_denominator)

        # 5. Sample w1
        w1 = np.random.uniform(-w1_max, w1_max)

        # 6. Calculate Coefficients (A, B, C)
        # From Eq 11: A*w2^2 + 2*B*w2 + C = 0
        A = m2 * (l2**2)
        B = m2 * l1 * l2 * w1 * np.cos(delta)
        C = M * (l1**2) * (w1**2) - 2 * (E - V)

        # 7. Solve for w2
        discriminant = B**2 - A * C
        
        # Numerical stability check: strictly, discriminant >= 0 due to w1_max calculation, 
        # but floating point errors might produce slightly negative numbers near zero.
        if discriminant < 0:
            discriminant = 0.0
            
        root = np.sqrt(discriminant)
        
        # Randomly select (+) or (-) solution
        sign = np.random.choice([-1.0, 1.0])
        w2 = (-B + sign * root) / A

        # 8. Store Valid State
        # Order: [theta_1, w1, theta_2, w2]
        inits.append((th1, w1, th2, w2))

    return inits


def main() -> None:
    params = PendulumParams()
    target_energy = 120.0  # Joules
    init_conditions = generate_initial_conditions(target_energy, 1_000, params)
    output_paths = [generate_traj(params, init_state) for init_state in init_conditions]
    print(f"\nSaved {len(output_paths)} trajectories.")
    # print(init_conditions)


if __name__ == "__main__":
    main()
