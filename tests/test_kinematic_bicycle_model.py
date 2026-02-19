import numpy as np
import pytest

from kinematic_bicycle_model import KinematicBicycleModel
from params import VehicleParams


@pytest.fixture
def vehicle_params() -> VehicleParams:
    return VehicleParams(
        wheelbase=2.0,
        delta_min=-0.6,
        delta_max=0.6,
        v_min=0.0,
        v_max=30.0,
        a_min=-5.0,
        a_max=3.0,
        steering_constant=0.5,
    )


@pytest.fixture
def model(vehicle_params: VehicleParams) -> KinematicBicycleModel:
    return KinematicBicycleModel(vehicle_params)


def test_continuous_dynamics_returns_expected_values(model: KinematicBicycleModel):
    state = np.array([1.0, 2.0, np.pi / 2.0, 4.0, 0.1])
    a = 1.5
    delta_cmd = 0.3

    out = model.continuous_dynamics(state, a, delta_cmd)

    expected = np.array([
        4.0 * np.cos(np.pi / 2.0),
        4.0 * np.sin(np.pi / 2.0),
        4.0 * np.tan(0.1) / 2.0,
        1.5,
        (0.3 - 0.1) / 0.5,
    ])
    assert np.allclose(out, expected)


def test_continuous_dynamics_raises_on_invalid_state_shape(model: KinematicBicycleModel):
    with pytest.raises(ValueError, match="state must be a 5-dimensional array"):
        model.continuous_dynamics(np.array([0.0, 1.0]), a=0.0, delta_cmd=0.0)
