import numpy as np
import pytest

from reference_trajectories import (
    circle_trajectory,
    clothoid_trajectory,
    figure_eight_trajectory,
    generate_trajectory,
    oval_trajectory,
    straight_line_trajectory,
)


def test_circle_trajectory_basic_properties():
    traj = circle_trajectory(n_points=200, dt=0.1, radius=10.0, speed=5.0)

    assert traj.x.shape == (200,)
    assert traj.y.shape == (200,)
    assert traj.psi.shape == (200,)
    assert traj.v.shape == (200,)
    assert traj.kappa.shape == (200,)
    assert np.allclose(traj.v, 5.0)
    assert np.allclose(traj.kappa, 0.1)

    r = np.sqrt(traj.x**2 + traj.y**2)
    assert np.allclose(r, 10.0, atol=1e-6)
    assert set(traj.initial_states.keys()) == {"easy", "medium", "hard"}
    assert traj.initial_states["easy"].shape == (2, 5)
    assert traj.initial_states["medium"].shape == (2, 5)
    assert traj.initial_states["hard"].shape == (2, 5)
    assert np.all(np.isfinite(traj.initial_states["easy"]))


def test_oval_trajectory_basic_properties():
    traj = oval_trajectory(n_points=300, dt=0.05, a=12.0, b=6.0, speed=4.0)

    assert traj.x.shape == (300,)
    assert traj.y.shape == (300,)
    assert np.max(np.abs(traj.x)) <= 12.0 + 1e-6
    assert np.max(np.abs(traj.y)) <= 6.0 + 1e-6
    assert np.all(np.isfinite(traj.kappa))


def test_figure_eight_crosses_origin_with_default_center():
    traj = figure_eight_trajectory(n_points=401, dt=0.05, a=8.0, speed=3.0)

    near_origin = (np.abs(traj.x) < 1e-6) & (np.abs(traj.y) < 1e-6)
    # Figure-8 parameterization should cross near the center multiple times.
    assert np.count_nonzero(near_origin) >= 2


def test_clothoid_curvature_is_linear_in_arc_length():
    traj = clothoid_trajectory(
        n_points=200,
        dt=0.1,
        length=40.0,
        speed=6.0,
        kappa_start=0.0,
        kappa_end=0.2,
    )

    assert np.isclose(traj.kappa[0], 0.0)
    assert np.isclose(traj.kappa[-1], 0.2)

    diff = np.diff(traj.kappa)
    assert np.allclose(diff, diff[0])


def test_generate_trajectory_dispatcher_and_state_conversion():
    traj = generate_trajectory(
        "circle",
        n_points=50,
        dt=0.1,
        radius=5.0,
        speed=2.0,
    )

    states = traj.as_state_array(wheelbase=2.5)
    assert states.shape == (50, 5)
    assert np.all(np.isfinite(states))


@pytest.mark.parametrize(
    ("direction", "expected_psi"),
    [
        ("pos_x", 0.0),
        ("neg_x", np.pi),
        ("pos_y", np.pi / 2.0),
        ("neg_y", -np.pi / 2.0),
    ],
)
def test_straight_line_trajectory_axis_aligned_directions(direction, expected_psi):
    traj = straight_line_trajectory(
        n_points=6,
        dt=0.2,
        speed=3.0,
        direction=direction,
    )

    assert np.isclose(traj.x[0], 0.0)
    assert np.isclose(traj.y[0], 0.0)
    assert np.allclose(traj.v, 3.0)
    assert np.allclose(traj.kappa, 0.0)
    assert np.allclose(traj.psi, expected_psi)

    if direction in ("pos_x", "neg_x"):
        assert np.allclose(traj.y, 0.0)
        if direction == "pos_x":
            assert np.all(np.diff(traj.x) >= 0.0)
        else:
            assert np.all(np.diff(traj.x) <= 0.0)
    else:
        assert np.allclose(traj.x, 0.0)
        if direction == "pos_y":
            assert np.all(np.diff(traj.y) >= 0.0)
        else:
            assert np.all(np.diff(traj.y) <= 0.0)


def test_generate_trajectory_dispatches_straight_line():
    traj = generate_trajectory(
        "straight_line",
        n_points=8,
        dt=0.1,
        speed=2.5,
        direction="neg_y",
    )
    assert np.allclose(traj.x, 0.0)
    assert np.all(np.diff(traj.y) <= 0.0)
    assert traj.initial_states["easy"].shape == (2, 5)
    assert traj.initial_states["medium"].shape == (2, 5)
    assert traj.initial_states["hard"].shape == (2, 5)


def test_generate_trajectory_raises_on_unknown_shape():
    with pytest.raises(ValueError, match="Unknown shape"):
        generate_trajectory("unknown", n_points=10, dt=0.1, radius=1.0, speed=1.0)
