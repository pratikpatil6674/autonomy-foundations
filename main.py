import numpy as np

from kinematic_bicycle_model import KinematicBicycleModel
from params import VehicleParams, SimParams, NoiseParams, NoiseMode
from simulation_plotter import SimulationPlotter
from vehicle_simulation import VehicleSimulation


veh_params = VehicleParams(
    wheelbase=2.7,
    delta_min=-np.pi/4,
    delta_max=np.pi/4,
    v_min=0,
    v_max=100,
    a_min=-6,
    a_max=4,
    steering_constant=0.25,
)

sim_params = SimParams(
    dt=0.1,
    integrator="rk4",
)

noise_params = NoiseParams(
    x_sigma=0.01,
    y_sigma=0.01,
    psi_sigma=0.01,
    v_sigma=0.01,
    delta_sigma=0.01,
    delta_cmd_sigma=0.01,
    a_sigma=0.1,
    steering_bias=0.0,
)

model = KinematicBicycleModel(veh_params)
simulation = VehicleSimulation(sim_params, noise_params, model)

# straight line
initial_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
delta_cmd_seq = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
a_seq = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# constant turn
initial_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
delta_cmd_seq = 0.5*np.ones(5)
a_seq = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

states = simulation.predict(initial_state, delta_cmd_seq, a_seq, NoiseMode.OFF)

print(states)

# Example selection string: "0,1" plots only state[0] and state[1] time-series.
selected_states = "0,1"
plotter = SimulationPlotter(dt=sim_params.dt, state_size=model.STATE_SIZE)
plotter.plot(
    states=states,
    state_indices=selected_states,
    delta_cmd_seq=delta_cmd_seq,
    a_seq=a_seq,
)
