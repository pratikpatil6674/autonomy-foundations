import numpy as np
from typing import Tuple
from kinematic_bicycle_model import KinematicBicycleModel
from params import NoiseMode, SimParams, NoiseParams

class VehicleSimulation:
    def __init__(
        self,
        sim_params: SimParams,
        noise_params: NoiseParams,
        model: KinematicBicycleModel,
    ):
        self.sim_params = sim_params  # simulation parameters
        self.noise_params = noise_params  # noise parameters
        self.model = model  # model
    
    def get_noisy_control(self, a: float, delta_cmd: float) -> Tuple[float, float]:
        """
        Get noisy control inputs.

        Args:
            a: acceleration
            delta_cmd: steering command
        """
        delta_cmd_noisy = delta_cmd + np.random.normal(0, self.noise_params.delta_cmd_sigma)
        a_noisy = a + np.random.normal(0, self.noise_params.a_sigma)
        return a_noisy, delta_cmd_noisy
    
    def get_noisy_state(self, state: np.ndarray) -> np.ndarray:
        """
        Get noisy state.

        Args:
            state: state
        """
        state_noisy = state + np.array([
            np.random.normal(0, self.noise_params.x_sigma),
            np.random.normal(0, self.noise_params.y_sigma),
            np.random.normal(0, self.noise_params.psi_sigma),
            np.random.normal(0, self.noise_params.v_sigma),
            np.random.normal(0, self.noise_params.delta_sigma),
        ])
        return state_noisy
    
    def step(
        self,
        state: np.ndarray,
        a: float,
        delta_cmd: float,
    ) -> np.ndarray:
        if self.sim_params.integrator == "euler":
            return self.model.continuous_dynamics(state, a, delta_cmd) * self.sim_params.dt + state
        elif self.sim_params.integrator == "rk4":
            k1 = self.model.continuous_dynamics(state, a, delta_cmd)
            k2 = self.model.continuous_dynamics(state + 0.5 * self.sim_params.dt * k1, a, delta_cmd)
            k3 = self.model.continuous_dynamics(state + 0.5 * self.sim_params.dt * k2, a, delta_cmd)
            k4 = self.model.continuous_dynamics(state + self.sim_params.dt * k3, a, delta_cmd)
            return state + (self.sim_params.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown integrator: {self.sim_params.integrator}")
    
    def clip_state(self, state: np.ndarray) -> np.ndarray:
        """
        Clip state to limits.

        Args:
            state: state
        """
        state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))   # wrap heading angle to [-pi, pi]
        state[3] = np.clip(state[3], self.model.params.v_min, self.model.params.v_max)
        state[4] = np.clip(state[4], self.model.params.delta_min, self.model.params.delta_max)
        return state
    
    def clip_controls(self, a: float, delta_cmd: float) -> Tuple[float, float]:
        """
        Clip controls to limits.

        Args:
            a: acceleration
            delta_cmd: steering command
        """
        delta_cmd = np.clip(delta_cmd, self.model.params.delta_min, self.model.params.delta_max)
        a = np.clip(a, self.model.params.a_min, self.model.params.a_max)
        return a, delta_cmd
    
    def predict(
        self,
        init_state: np.ndarray,
        delta_cmd_seq: np.ndarray,
        a_seq: np.ndarray,
        noise_mode: NoiseMode
    ) -> np.ndarray:
        """
        Predict trajectory over a sequence of control inputs.

        Args:
            state: initial state
            delta_cmd_seq: sequence of steering commands
            a_seq: sequence of accelerations

        Returns:
            states: (N+1) x 5 array (includes initial state)
        """
        if len(delta_cmd_seq) != len(a_seq):
            raise ValueError(f"delta_cmd_seq and a_seq must have same length, got {len(delta_cmd_seq)} and {len(a_seq)}")
        if len(delta_cmd_seq) == 0:
            raise ValueError("delta_cmd_seq and a_seq must have at least one element")
        n = len(delta_cmd_seq)
        states = np.zeros((n + 1, self.model.STATE_SIZE))
        states[0] = init_state
        for i in range(n):
            a = a_seq[i]
            delta_cmd = delta_cmd_seq[i]
            if noise_mode != NoiseMode.OFF:
                a, delta_cmd = self.get_noisy_control(a, delta_cmd)
            a, delta_cmd = self.clip_controls(a, delta_cmd)

            states[i+1] = self.step(states[i], a, delta_cmd) 
            if noise_mode != NoiseMode.OFF:
                states[i+1] = self.get_noisy_state(states[i+1]) 
            states[i+1] = self.clip_state(states[i+1])
        return states
