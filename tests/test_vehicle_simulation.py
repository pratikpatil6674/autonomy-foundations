import numpy as np
import pytest

from kinematic_bicycle_model import KinematicBicycleModel
from params import NoiseMode, NoiseParams, SimParams, VehicleParams
from vehicle_simulation import VehicleSimulation


@pytest.fixture
def vehicle_params() -> VehicleParams:
    return VehicleParams(
        wheelbase=2.0,
        delta_min=-0.4,
        delta_max=0.4,
        v_min=0.0,
        v_max=20.0,
        a_min=-2.0,
        a_max=2.0,
        steering_constant=0.25,
    )


@pytest.fixture
def noise_params_zero() -> NoiseParams:
    return NoiseParams(
        x_sigma=0.0,
        y_sigma=0.0,
        psi_sigma=0.0,
        v_sigma=0.0,
        delta_sigma=0.0,
        delta_cmd_sigma=0.0,
        a_sigma=0.0,
        steering_bias=0.0,
    )


@pytest.fixture
def model(vehicle_params: VehicleParams) -> KinematicBicycleModel:
    return KinematicBicycleModel(vehicle_params)


def make_sim(sim_params: SimParams, noise_params: NoiseParams, model: KinematicBicycleModel) -> VehicleSimulation:
    return VehicleSimulation(sim_params=sim_params, noise_params=noise_params, model=model)


def test_clip_controls_clips_both_inputs(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="euler"), noise_params_zero, model)

    a, delta_cmd = sim.clip_controls(a=8.0, delta_cmd=-2.0)

    assert a == model.params.a_max
    assert delta_cmd == model.params.delta_min


def test_clip_state_wraps_heading_and_clips_v_delta(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="euler"), noise_params_zero, model)
    state = np.array([0.0, 0.0, 4.0 * np.pi, 100.0, -10.0])

    clipped = sim.clip_state(state)

    assert np.isclose(clipped[2], 0.0)
    assert clipped[3] == model.params.v_max
    assert clipped[4] == model.params.delta_min


def test_step_euler_straight_motion(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="euler"), noise_params_zero, model)
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0])

    next_state = sim.step(state, a=0.0, delta_cmd=0.0)

    expected = np.array([1.0, 0.0, 0.0, 10.0, 0.0])
    assert np.allclose(next_state, expected)


def test_step_rk4_straight_motion(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="rk4"), noise_params_zero, model)
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0])

    next_state = sim.step(state, a=0.0, delta_cmd=0.0)

    expected = np.array([1.0, 0.0, 0.0, 10.0, 0.0])
    assert np.allclose(next_state, expected)


def test_predict_returns_expected_shape_and_initial_state(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="euler"), noise_params_zero, model)
    state0 = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    delta_cmd_seq = np.array([0.0, 0.0, 0.0])
    a_seq = np.array([0.0, 0.0, 0.0])

    states = sim.predict(init_state=state0, delta_cmd_seq=delta_cmd_seq, a_seq=a_seq, noise_mode=NoiseMode.OFF)

    assert states.shape == (4, model.STATE_SIZE)
    assert np.allclose(states[0], state0)


def test_predict_raises_for_mismatched_control_lengths(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="euler"), noise_params_zero, model)
    with pytest.raises(ValueError, match="must have same length"):
        sim.predict(
            init_state=np.zeros(model.STATE_SIZE),
            delta_cmd_seq=np.array([0.1, 0.2]),
            a_seq=np.array([0.0]),
            noise_mode=NoiseMode.OFF,
        )


def test_predict_raises_for_empty_controls(model: KinematicBicycleModel, noise_params_zero: NoiseParams):
    sim = make_sim(SimParams(dt=0.1, integrator="euler"), noise_params_zero, model)
    with pytest.raises(ValueError, match="at least one element"):
        sim.predict(
            init_state=np.zeros(model.STATE_SIZE),
            delta_cmd_seq=np.array([]),
            a_seq=np.array([]),
            noise_mode=NoiseMode.OFF,
        )
