from __future__ import annotations

import math
from dataclasses import dataclass
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
    t_stop: float = 100.0
    dt: float = 0.01

    @property
    def total_length(self) -> float:
        return self.L1 + self.L2


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
    output_dir = Path("paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def generate_traj(params: PendulumParams, init_state: Sequence[float], verbose = True) -> Path:
    """Generate a trajectory CSV containing theta1/theta2 columns."""
    _, traj = integrate_trajectory(params, init_state)
    angles = traj[:, [0, 2]]
    angles = ((angles + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]
    output_path = _build_output_path(normalize_init_state(init_state))
    np.savetxt(output_path, angles, delimiter=",", header="theta1,theta2", comments="")
    if verbose:
        print(f"Saved trajectory to {output_path}")
    return output_path


def generate_initial_conditions(
    E: float,
    n: int,
    params: PendulumParams,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float, float, float]]:
    """Generate n initial (theta_deg, w1, theta_deg, w2) tuples near the energy surface."""
    rng = rng or np.random.default_rng()
    inits: list[tuple[float, float, float, float]] = []

    m1, m2 = params.M1, params.M2
    l1, l2 = params.L1, params.L2
    g = params.G

    attempts = 0
    max_attempts = n * 50

    while len(inits) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not generate {n} initial conditions within {max_attempts} attempts.")

        th1 = rng.uniform(0, 2 * np.pi)
        th2 = rng.uniform(0, 2 * np.pi)

        V = -(m1 + m2) * g * l1 * math.cos(th1) - m2 * g * l2 * math.cos(th2)
        KE_avail = E - V
        if KE_avail <= 0:
            continue

        r = rng.random()
        KE1 = r * KE_avail
        KE2 = (1 - r) * KE_avail

        w1 = math.sqrt(2 * KE1 / ((m1 + m2) * l1 * l1))
        w2 = math.sqrt(2 * KE2 / (m2 * l2 * l2))

        w1 = math.copysign(w1, rng.normal())
        w2 = math.copysign(w2, rng.normal())

        inits.append((math.degrees(th1), w1, math.degrees(th2), w2))

    return inits


def main() -> None:
    params = PendulumParams()
    target_energy = 5.0  # Joules
    init_conditions = generate_initial_conditions(target_energy, 1_000, params)
    output_paths = [generate_traj(params, init_state) for init_state in init_conditions]
    print(f"\nSaved {len(output_paths)} trajectories to paths/")
    # print(init_conditions)


if __name__ == "__main__":
    main()
