import numpy as np
from typing import Tuple, Optional
import sympy as sp

from params import VehicleParams

class KinematicBicycleModel:
    def __init__(self, params: VehicleParams):
        self.params = params
        self.STATE_SIZE = 5 # x, y, psi, v, delta
    
    def continuous_dynamics(
        self,
        state: np.ndarray,
        a: float,
        delta_cmd: float
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
                a: acceleration (m/s^2)
                delta_cmd: steering command (rad)

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
    
    def jacobians(self) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute the Jacobians of the continuous dynamics.
        
        Returns:
            A: Jacobian with respect to state
            B: Jacobian with respect to input
        """
        x, y, psi, v, delta = sp.symbols('x y psi v delta', real=True)
        a, delta_cmd, L, dt, tau = sp.symbols('a delta_cmd L dt tau', real=True)

        X = sp.Matrix([x, y, psi, v, delta])
        U = sp.Matrix([a, delta_cmd])

        f = sp.Matrix([
            x + v*sp.cos(psi)*dt,
            y + v*sp.sin(psi)*dt,
            psi + (v/L)*sp.tan(delta)*dt,
            v + a*dt,
            delta + (delta_cmd - delta)*dt / tau
        ])

        A = f.jacobian(X)  # df/dx
        B = f.jacobian(U)  # df/du

        A_num = sp.lambdify((x, y, psi, v, delta, a, delta_cmd, L, tau, dt), A, "numpy")
        B_num = sp.lambdify((x, y, psi, v, delta, a, delta_cmd, L, tau, dt), B, "numpy")

        return A_num, B_num
    
