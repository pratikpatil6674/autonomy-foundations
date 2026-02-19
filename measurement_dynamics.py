
import numpy as np

class MeasurementDynamics:
    def __init__(self, noise_params):
        self.noise_params = noise_params
    
    def measurement(self, state: np.ndarray) -> np.ndarray:
        """
        Measurement function.
        
        Args:
            state: state
            
        Returns:
            measurement: measurement
        """
        return np.array([state[0], state[1]])
    
    def get_noisy_measurement(self, state: np.ndarray) -> np.ndarray:
        """
        Get noisy measurement.
        
        Args:
            state: state
            
        Returns:
            noisy_measurement: noisy measurement
        """
        return np.array([state[0] + np.random.normal(0, self.noise_params.x_sigma), state[1] + np.random.normal(0, self.noise_params.y_sigma)])
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Jacobian of the measurement function.
        
        Args:
            state: state
            
        Returns:
            jacobian: jacobian
        """
        return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])