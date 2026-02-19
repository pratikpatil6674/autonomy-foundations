
import numpy as np
class KalmanFilter:
    def __init__(
        self, 
        state_dim: int, 
        input_dim: int, 
        measurement_dim: int,
        x_init: np.ndarray,
        P_init: np.ndarray,
        normalize_state: callable = None,
        normalize_measurement: callable = None
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.measurement_dim = measurement_dim

        self.x_pred = np.zeros(self.state_dim) # predicted state
        self.P_pred = np.zeros((self.state_dim, self.state_dim)) # predicted error covariance

        self.y = np.zeros(self.measurement_dim)    # innovation
        self.S = np.zeros((self.measurement_dim, self.measurement_dim)) # innovation covariance
        self.K = np.zeros((self.state_dim, self.measurement_dim)) # kalman gain
        self.x_init = x_init
        self.x_post = x_init # updated state
        self.P_post = P_init # updated error covariance
    
        self.normalize_state = normalize_state
        self.normalize_measurement = normalize_measurement
    
    def predict(self, u, f, F, Q):
        """
        Predict the next state and error covariance.
        f: state transition function
        F: state transition jacobian
        Q: process noise covariance
        """
        self.x_pred = f(self.x_post, *u)
        self.P_pred = F @ self.P_post @ F.T + Q
        
        if self.normalize_state is not None:
            self.x_pred = self.normalize_state(self.x_pred)
        
        return self.x_pred, self.P_pred
        
    def update(self, z, h, H, R):
        """
        Update the state and error covariance.
        h: measurement function
        H: measurement jacobian
        R: measurement noise covariance
        """
        self.y = z - h(self.x_pred) # innovation

        if self.normalize_measurement is not None:
            self.y = self.normalize_measurement(self.y)
        
        self.S = H @ self.P_pred @ H.T + R # innovation covariance
        self.K = self.P_pred @ H.T @ np.linalg.inv(self.S) # kalman gain

        self.x_post = self.x_pred + self.K @ self.y
        self.P_post = (np.eye(self.state_dim) - self.K @ H) @ self.P_pred @ \
                      (np.eye(self.state_dim) - self.K @ H).T + self.K @ R @ self.K.T

        if self.normalize_state is not None:
            self.x_post = self.normalize_state(self.x_post)
        
        return self.x_post, self.P_post