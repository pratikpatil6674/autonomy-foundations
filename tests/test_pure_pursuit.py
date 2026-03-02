import numpy as np
import pytest
import os

from kinematic_bicycle_model import KinematicBicycleModel
from params import NoiseMode, NoiseParams, SimParams, VehicleParams
from vehicle_simulation import VehicleSimulation
from measurement_dynamics import MeasurementDynamics
from simulation_plotter import SimulationPlotter
import reference_trajectories as rt
from pure_pursuit import PurePursuit

@pytest.fixture
def vehicle_params() -> VehicleParams:
    return VehicleParams(
        wheelbase=2.0,
        delta_min=-0.8,
        delta_max=0.8,
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

@pytest.fixture
def pure_pursuit_controller(vehicle_params):
    return PurePursuit(vehicle_params, Lmin=0.5, k=0.5)

def test_pure_pursuit_tracks_path(
    vehicle_params, noise_params_nonzero, pure_pursuit_controller
):
    np.random.seed(0)
    vh = VehicleSimulation(
        sim_params=SimParams(dt=0.1, integrator="euler"),
        noise_params=noise_params_nonzero,
        model=KinematicBicycleModel(vehicle_params)
    )

    # Generate reference trajectory
    reference_trajectory = rt.circle_trajectory(
        n_points=50,
        dt=0.1,
        radius=10.0,
        speed=5.0,
        center=(0.0, 0.0),
        laps=1.0,
        phase=0.0,
    )
    # reference_trajectory = rt.oval_trajectory(
    #     n_points=50,
    #     dt=0.1,
    #     a=10.0,
    #     b=7.0,
    #     speed=1.0,
    #     center=(0.0, 0.0),
    #     laps=1.0,
    #     phase=0.0,
    # )
    # reference_trajectory = rt.figure_eight_trajectory(
    #     n_points=50,
    #     dt=0.1,
    #     a=10.0,
    #     speed=1.0,
    #     center=(0.0, 0.0),
    #     laps=1.0,
    #     phase=0.0,
    # )
    states_true = reference_trajectory.as_state_array(vehicle_params.wheelbase)
    state = np.array([0.0, 10.0, -np.pi, 5.0, 0.0])
    states = []
    delta_cmds = []
    a = 0.0
    i = 0
    while True:
        delta_cmd = pure_pursuit_controller.get_steering_command(state, reference_trajectory)
        state = vh.step(state, a, delta_cmd, clip_state=True)
        states.append(state)
        delta_cmds.append(delta_cmd)
        i += 1
        if i >= 500:
            break
    
    plotter = SimulationPlotter(dt=0.1, state_size=5)
    show_plots = os.getenv("SHOW_PLOTS", "0") == "1"
    plotter.plot(
        states=[states_true[1:len(states) + 1], states],
        state_indices="0,1,2",
        delta_cmd_seq=[delta_cmds],
        experiment_labels=["true", "pure_pursuit"],
        show=show_plots,
    )
