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


def generate_traj(params: PendulumParams, init_state: Sequence[float]) -> Path:
    """Generate a trajectory CSV containing theta1/theta2 columns."""
    _, traj = integrate_trajectory(params, init_state)
    angles = traj[:, [0, 2]]
    angles = ((angles + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]
    output_path = _build_output_path(normalize_init_state(init_state))
    np.savetxt(output_path, angles, delimiter=",", header="theta1,theta2", comments="")
    return output_path


def main() -> None:
    params = PendulumParams()
    for theta1 in range(0, 360, 10):
        for theta2 in range(0, 360, 10):
            init_conditions = (theta1, 0.0, theta2, 0.0)
            output_path = generate_traj(params, init_conditions)
            print(f"Saved trajectory to {output_path}")


if __name__ == "__main__":
    main()
