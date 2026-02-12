import numpy as np
from typing import Tuple, Optional


from params import VehicleParams

class KinematicBicycleModel:
    def __init__(self, params: VehicleParams):
        self.params = params
        self.STATE_SIZE = 5 # x, y, psi, v, delta
    
    def continuous_dynamics(
        self,
        state: np.ndarray,
        delta_cmd: float,
        a: float
    ) -> np.ndarray:
        """
        Continuous dynamics of the kinematic bicycle model.

        Args:
            state: state
                x: x position
                y: y position
                psi: heading angle
                v: velocity
                delta: steering angle
            delta_cmd: steering command (rad)
            a: acceleration (m/s^2)

        Returns:
            state: next state
        """
        if state.shape != (self.STATE_SIZE,):
            raise ValueError(f"state must be a {self.STATE_SIZE}-dimensional array, got {state.shape}")
        x, y, psi, v, delta = state
        return np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            v * np.tan(delta) / self.params.wheelbase,
            a,
            (delta_cmd - delta) / self.params.steering_constant,
        ])
    
    