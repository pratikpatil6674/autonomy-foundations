import numpy as np
import pytest
import os

from kinematic_bicycle_model import KinematicBicycleModel
from params import NoiseMode, NoiseParams, SimParams, VehicleParams
from vehicle_simulation import VehicleSimulation
from measurement_dynamics import MeasurementDynamics
from simulation_plotter import SimulationPlotter
import reference_trajectories as rt
import mpc
@pytest.fixture
def vehicle_params() -> VehicleParams:
    return VehicleParams(
        wheelbase=2.0,
        delta_min=-0.8,
        delta_max=0.8,
        v_min=0.0,
        v_max=100.0,
        a_min=-5.0,
        a_max=5.0,
        steering_constant=0.2,
    )

@pytest.fixture
def sim_params() -> SimParams:
    return SimParams(dt=0.01, integrator="euler")

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


def test_mpc(
    vehicle_params, noise_params_nonzero, sim_params
):
    np.random.seed(0)
    vh = VehicleSimulation(
        sim_params=sim_params,
        noise_params=noise_params_nonzero,
        model=KinematicBicycleModel(vehicle_params)
    )

    # Generate reference trajectory
    reference_trajectory = rt.circle_trajectory(
        n_points=400,
        dt=sim_params.dt,
        radius=10.0,
        speed=0.1,
        center=(0.0, 0.0),
        laps=1.0,
        phase=0.0,
        wheelbase=vehicle_params.wheelbase,
    )
    # reference_trajectory = rt.straight_line_trajectory(
    #     n_points=400,
    #     dt=sim_params.dt,
    #     speed=5.0,
    #     direction="pos_x",
    #     wheelbase=vehicle_params.wheelbase,
    # )
    # reference_trajectory = rt.oval_trajectory(
    #     n_points=800,
    #     dt=sim_params.dt,
    #     a=10.0,
    #     b=7.0,
    #     speed=1.0,
    #     center=(0.0, 0.0),
    #     laps=2.0,
    #     phase=0.0,
    # )
    reference_trajectory = rt.figure_eight_trajectory(
        n_points=800,
        dt=sim_params.dt,
        a=10.0,
        speed=0.1,
        center=(0.0, 0.0),
        laps=2.0,
        phase=0.0,
    )

    mpc_controller = mpc.MPC(vehicle_params, sim_params, reference_trajectory)
    mpc_controller.build_simulator()
    
    states_true = reference_trajectory.as_state_array(vehicle_params.wheelbase)
    state_initial = reference_trajectory.initial_states["hard"][1]
    state_initial = states_true[0]
    state_initial[3] = 15.0
    # state_initial[0] = 30.0
 
    states, inputs = mpc_controller.run_closed_loop(state_initial, steps=800)
    states = np.array(states)
    states = states.squeeze()
    print(states.shape)
    print(states[0].shape)
    print(states[1].shape)
    
    print(states_true.shape)
    print(states_true[0].shape)
    print(states_true[1].shape)

    print(type(inputs))
    breakpoint()
    inputs = np.array(inputs)
    plotter = SimulationPlotter(dt=sim_params.dt, state_size=5)
    show_plots = os.getenv("SHOW_PLOTS", "0") == "1"
    animate_plots = os.getenv("ANIMATE_PLOTS", "0") == "1"
    plotter.plot(
        states=[states_true[1:len(states) + 1], states],
        delta_cmd_seq=inputs[:, 1].reshape(-1),
        a_seq=inputs[:, 0].reshape(-1),
        state_indices="0,1,2,3,4",
        experiment_labels=["true", "mpc"],
        show=show_plots,
        animate=animate_plots,
        animate_heading_length=1.8
    )
