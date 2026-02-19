import numpy as np
import pytest
import os

from kinematic_bicycle_model import KinematicBicycleModel
from params import NoiseMode, NoiseParams, SimParams, VehicleParams
from vehicle_simulation import VehicleSimulation
from measurement_dynamics import MeasurementDynamics
from kalman_filter import KalmanFilter
from simulation_plotter import SimulationPlotter

@pytest.fixture
def kalman_filter():
    return KalmanFilter(
        state_dim=5,
        input_dim=2,
        measurement_dim=2,
        x_init=np.array([0.0, 0.0, 0.0, 5.0, 0.0]),
        P_init=np.eye(5)*0.01,
        normalize_state=None,
        normalize_measurement=None
    )
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
def noise_params_nonzero() -> NoiseParams:
    return NoiseParams(
        x_sigma=0.1,
        y_sigma=0.1,
        psi_sigma=0.1,
        v_sigma=0.1,
        delta_sigma=0.1,
        delta_cmd_sigma=0.0,
        a_sigma=0.0,
        steering_bias=0.0,
    )

def test_kalman_filter_tracks_xy_better_than_noisy_measurements(
    kalman_filter, vehicle_params, noise_params_nonzero
):
    np.random.seed(0)
    delta_cmd_seq = 0.5*np.ones(50)
    a_seq = np.ones(50) * 0.5
    vh = VehicleSimulation(
        sim_params=SimParams(dt=0.1, integrator="euler"),
        noise_params=noise_params_nonzero,
        model=KinematicBicycleModel(vehicle_params)
    )
    states_true = vh.predict(kalman_filter.x_init, delta_cmd_seq, a_seq, NoiseMode.IID)

    m = MeasurementDynamics(noise_params_nonzero)
    h = m.measurement
    H = m.jacobian(kalman_filter.x_post)

    f = vh.step
    A, _ = vh.model.jacobians()

    Q = np.eye(5) * 0.01 # process noise covariance
    R = np.eye(2) * 0.01 # measurement noise covariance

    states_pred = []
    states = []
    measurements = []
    for i in range(len(delta_cmd_seq)):
        # if i in [20, 21, 22, 23, 24]:
        #     a_seq[i] = 15.0
        #     delta_cmd_seq[i] = 1.0
        u = np.array([a_seq[i], delta_cmd_seq[i]])
        A_val = A(*kalman_filter.x_post, *u, vh.model.params.wheelbase, vh.model.params.steering_constant, vh.sim_params.dt)
        x_pred, _ = kalman_filter.predict(u, f, A_val, Q)
        states_pred.append(x_pred)
        z = m.get_noisy_measurement(states_true[i+1])
        measurements.append(z)
        x_post, _ = kalman_filter.update(z, h, H, R)
        states.append(x_post)

    states_pred = np.array(states_pred)
    states = np.array(states)
    measurements = np.array(measurements)
    states_true_step = states_true[1:len(states) + 1]

    # Sanity checks on shape and finite values.
    assert states_pred.shape == (len(delta_cmd_seq), 5)
    assert states.shape == (len(delta_cmd_seq), 5)
    assert measurements.shape == (len(delta_cmd_seq), 2)
    assert np.all(np.isfinite(states_pred))
    assert np.all(np.isfinite(states))

    rmse_pred_xy = np.sqrt(np.mean((states_pred[:, :2] - states_true_step[:, :2]) ** 2))
    rmse_post_xy = np.sqrt(np.mean((states[:, :2] - states_true_step[:, :2]) ** 2))

    # Measurement update should improve over pure model prediction.
    assert rmse_post_xy < rmse_pred_xy

    # Posterior covariance should stay symmetric positive semi-definite.
    assert np.allclose(kalman_filter.P_post, kalman_filter.P_post.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(kalman_filter.P_post)) >= -1e-10

    plotter = SimulationPlotter(dt=0.1, state_size=5)
    show_plots = os.getenv("SHOW_PLOTS", "0") == "1"
    plotter.plot(
        states=[states_true[1:len(states) + 1], states],
        state_indices="0,1",
        a_seq=a_seq,
        experiment_labels=["true", "kf"],
        show=show_plots,
    )
