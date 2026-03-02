from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np


@dataclass
class ReferenceTrajectory:
    """Container for a time-parameterized planar reference trajectory."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    kappa: np.ndarray
    delta: np.ndarray
    initial_states: Dict[str, np.ndarray]

    def as_state_array(self, wheelbase: Optional[float] = None) -> np.ndarray:
        """
        Convert trajectory to kinematic bicycle state array [x, y, psi, v, delta].

        Args:
            wheelbase:
                Optional vehicle wheelbase used to recompute steering from `kappa`.
                If `None`, use stored `delta` values from this trajectory.

        Returns:
            Array with shape (N, 5).
        """
        if wheelbase is None:
            delta = self.delta
        else:
            delta = np.arctan(wheelbase * self.kappa)
        return np.column_stack([self.x, self.y, self.psi, self.v, delta])


def _normalize_speed(speed: float, n: int) -> np.ndarray:
    if speed <= 0.0:
        raise ValueError(f"speed must be > 0, got {speed}")
    return np.full(n, float(speed))


def _ensure_n_points(n_points: int) -> None:
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")


def _delta_from_kappa(kappa: np.ndarray, wheelbase: float) -> np.ndarray:
    if wheelbase <= 0.0:
        raise ValueError(f"wheelbase must be > 0, got {wheelbase}")
    return np.arctan(wheelbase * kappa)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _build_initial_states(
    x0: float,
    y0: float,
    psi0: float,
    v0: float,
    delta0: float,
) -> Dict[str, np.ndarray]:
    """
    Build six curated initial states grouped by difficulty.

    Returns:
        Dict with keys `easy`, `medium`, `hard`, each value shape (2, 5)
        in `[x, y, psi, v, delta]` format.
    """
    heading = np.array([np.cos(psi0), np.sin(psi0)])
    left = np.array([-np.sin(psi0), np.cos(psi0)])
    base_pos = np.array([x0, y0])

    def _state(
        along: float,
        lateral: float,
        dpsi: float,
        v_scale: float,
        ddelta: float,
    ) -> np.ndarray:
        pos = base_pos + along * heading + lateral * left
        return np.array(
            [
                pos[0],
                pos[1],
                _wrap_angle(psi0 + dpsi),
                max(0.05, v0 * v_scale),
                delta0 + ddelta,
            ]
        )

    return {
        "easy": np.array(
            [
                _state(along=0.0, lateral=0.0, dpsi=0.0, v_scale=1.0, ddelta=0.0),
                _state(along=0.3, lateral=0.15, dpsi=0.05, v_scale=0.9, ddelta=0.02),
            ]
        ),
        "medium": np.array(
            [
                _state(along=-1.0, lateral=0.9, dpsi=0.25, v_scale=0.7, ddelta=-0.08),
                _state(along=1.0, lateral=-0.9, dpsi=-0.25, v_scale=1.3, ddelta=0.08),
            ]
        ),
        "hard": np.array(
            [
                _state(along=-3.0, lateral=2.5, dpsi=0.75, v_scale=0.3, ddelta=-0.2),
                _state(along=3.0, lateral=-2.5, dpsi=-0.75, v_scale=1.8, ddelta=0.2),
            ]
        ),
    }


def circle_trajectory(
    n_points: int,
    dt: float,
    radius: float,
    speed: float,
    center: Tuple[float, float] = (0.0, 0.0),
    laps: float = 1.0,
    phase: float = 0.0,
    wheelbase: float = 1.0,
) -> ReferenceTrajectory:
    """Generate a circle trajectory."""
    _ensure_n_points(n_points)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if laps <= 0.0:
        raise ValueError(f"laps must be > 0, got {laps}")

    theta = np.linspace(phase, phase + 2.0 * np.pi * laps, n_points)
    cx, cy = center

    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    psi = theta + np.pi / 2.0
    v = _normalize_speed(speed, n_points)
    kappa = np.full(n_points, 1.0 / radius)
    delta = _delta_from_kappa(kappa, wheelbase)
    t = np.arange(n_points) * dt
    initial_states = _build_initial_states(x[0], y[0], psi[0], v[0], delta[0])

    return ReferenceTrajectory(
        t=t, x=x, y=y, psi=psi, v=v, kappa=kappa, delta=delta, initial_states=initial_states
    )


def oval_trajectory(
    n_points: int,
    dt: float,
    a: float,
    b: float,
    speed: float,
    center: Tuple[float, float] = (0.0, 0.0),
    laps: float = 1.0,
    phase: float = 0.0,
    wheelbase: float = 1.0,
) -> ReferenceTrajectory:
    """Generate an oval (ellipse) trajectory."""
    _ensure_n_points(n_points)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be > 0, got a={a}, b={b}")
    if laps <= 0.0:
        raise ValueError(f"laps must be > 0, got {laps}")

    theta = np.linspace(phase, phase + 2.0 * np.pi * laps, n_points)
    cx, cy = center

    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)

    dx_dtheta = -a * np.sin(theta)
    dy_dtheta = b * np.cos(theta)
    psi = np.arctan2(dy_dtheta, dx_dtheta)

    ddx_dtheta = -a * np.cos(theta)
    ddy_dtheta = -b * np.sin(theta)
    denom = (dx_dtheta**2 + dy_dtheta**2) ** 1.5
    # Avoid division by zero for degenerate parameterizations (shouldn't happen for a,b>0).
    denom = np.where(denom < 1e-12, 1e-12, denom)
    kappa = (dx_dtheta * ddy_dtheta - dy_dtheta * ddx_dtheta) / denom

    v = _normalize_speed(speed, n_points)
    delta = _delta_from_kappa(kappa, wheelbase)
    t = np.arange(n_points) * dt
    initial_states = _build_initial_states(x[0], y[0], psi[0], v[0], delta[0])

    return ReferenceTrajectory(
        t=t, x=x, y=y, psi=psi, v=v, kappa=kappa, delta=delta, initial_states=initial_states
    )


def figure_eight_trajectory(
    n_points: int,
    dt: float,
    a: float,
    speed: float,
    center: Tuple[float, float] = (0.0, 0.0),
    laps: float = 1.0,
    phase: float = 0.0,
    wheelbase: float = 1.0,
) -> ReferenceTrajectory:
    """
    Generate a figure-8 trajectory using a Lissajous-style parameterization.

    x = a * sin(theta), y = (a/2) * sin(2*theta)
    """
    _ensure_n_points(n_points)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if a <= 0.0:
        raise ValueError(f"a must be > 0, got {a}")
    if laps <= 0.0:
        raise ValueError(f"laps must be > 0, got {laps}")

    theta = np.linspace(phase, phase + 2.0 * np.pi * laps, n_points)
    cx, cy = center

    x = cx + a * np.sin(theta)
    y = cy + 0.5 * a * np.sin(2.0 * theta)

    dx_dtheta = a * np.cos(theta)
    dy_dtheta = a * np.cos(2.0 * theta)
    psi = np.arctan2(dy_dtheta, dx_dtheta)

    ddx_dtheta = -a * np.sin(theta)
    ddy_dtheta = -2.0 * a * np.sin(2.0 * theta)
    denom = (dx_dtheta**2 + dy_dtheta**2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    kappa = (dx_dtheta * ddy_dtheta - dy_dtheta * ddx_dtheta) / denom

    v = _normalize_speed(speed, n_points)
    delta = _delta_from_kappa(kappa, wheelbase)
    t = np.arange(n_points) * dt
    initial_states = _build_initial_states(x[0], y[0], psi[0], v[0], delta[0])

    return ReferenceTrajectory(
        t=t, x=x, y=y, psi=psi, v=v, kappa=kappa, delta=delta, initial_states=initial_states
    )


def clothoid_trajectory(
    n_points: int,
    dt: float,
    length: float,
    speed: float,
    kappa_start: float = 0.0,
    kappa_end: float = 0.1,
    x0: float = 0.0,
    y0: float = 0.0,
    psi0: float = 0.0,
    wheelbase: float = 1.0,
) -> ReferenceTrajectory:
    """
    Generate a clothoid (Euler spiral) where curvature varies linearly with arc length.
    """
    _ensure_n_points(n_points)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")

    s = np.linspace(0.0, length, n_points)
    ds = s[1] - s[0]

    kappa = kappa_start + (kappa_end - kappa_start) * (s / length)
    dk_ds = (kappa_end - kappa_start) / length

    psi = psi0 + kappa_start * s + 0.5 * dk_ds * s**2

    x = np.zeros(n_points)
    y = np.zeros(n_points)
    x[0] = x0
    y[0] = y0
    for i in range(1, n_points):
        x[i] = x[i - 1] + ds * np.cos(psi[i - 1])
        y[i] = y[i - 1] + ds * np.sin(psi[i - 1])

    v = _normalize_speed(speed, n_points)
    delta = _delta_from_kappa(kappa, wheelbase)
    t = np.arange(n_points) * dt
    initial_states = _build_initial_states(x[0], y[0], psi[0], v[0], delta[0])

    return ReferenceTrajectory(
        t=t, x=x, y=y, psi=psi, v=v, kappa=kappa, delta=delta, initial_states=initial_states
    )


def straight_line_trajectory(
    n_points: int,
    dt: float,
    speed: float,
    direction: Literal["pos_x", "neg_x", "pos_y", "neg_y"] = "pos_x",
    wheelbase: float = 1.0,
) -> ReferenceTrajectory:
    """
    Generate an axis-aligned straight-line trajectory starting at the origin.

    Args:
        n_points: Number of trajectory samples.
        dt: Sampling period [s].
        speed: Constant speed magnitude [m/s].
        direction: One of `pos_x`, `neg_x`, `pos_y`, `neg_y`.

    Returns:
        ReferenceTrajectory sampled in time.
    """
    _ensure_n_points(n_points)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    v = _normalize_speed(speed, n_points)
    t = np.arange(n_points) * dt
    distance = speed * t

    x = np.zeros(n_points)
    y = np.zeros(n_points)
    if direction == "pos_x":
        x = distance
        psi = np.zeros(n_points)
    elif direction == "neg_x":
        x = -distance
        psi = np.full(n_points, np.pi)
    elif direction == "pos_y":
        y = distance
        psi = np.full(n_points, np.pi / 2.0)
    elif direction == "neg_y":
        y = -distance
        psi = np.full(n_points, -np.pi / 2.0)
    else:
        raise ValueError(
            f"Unknown direction '{direction}'. Expected one of: pos_x, neg_x, pos_y, neg_y"
        )

    kappa = np.zeros(n_points)
    delta = _delta_from_kappa(kappa, wheelbase)
    initial_states = _build_initial_states(x[0], y[0], psi[0], v[0], delta[0])
    return ReferenceTrajectory(
        t=t, x=x, y=y, psi=psi, v=v, kappa=kappa, delta=delta, initial_states=initial_states
    )


def generate_trajectory(
    shape: Literal["circle", "oval", "figure_8", "clothoid", "straight_line"],
    **kwargs,
) -> ReferenceTrajectory:
    """Dispatcher for trajectory generation by shape name."""
    if shape == "circle":
        return circle_trajectory(**kwargs)
    if shape == "oval":
        return oval_trajectory(**kwargs)
    if shape == "figure_8":
        return figure_eight_trajectory(**kwargs)
    if shape == "clothoid":
        return clothoid_trajectory(**kwargs)
    if shape == "straight_line":
        return straight_line_trajectory(**kwargs)
    raise ValueError(
        f"Unknown shape '{shape}'. Expected one of: circle, oval, figure_8, clothoid, straight_line"
    )

def sample_reference_trajectory(trajectory: ReferenceTrajectory, t: float) -> np.ndarray:
    """Sample a reference trajectory at a given time."""
    t_clamped = np.clip(t, trajectory.t[0], trajectory.t[-1])
    Xr = np.interp(t_clamped, trajectory.t, trajectory.x)
    Yr = np.interp(t_clamped, trajectory.t, trajectory.y)
    Psir = np.interp(t_clamped, trajectory.t, trajectory.psi)
    Vr = np.interp(t_clamped, trajectory.t, trajectory.v)
    Deltar = np.interp(t_clamped, trajectory.t, trajectory.delta)

    return np.array([Xr, Yr, Psir, Vr, Deltar])
