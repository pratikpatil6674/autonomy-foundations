import numpy as np
from reference_trajectories import ReferenceTrajectory
from params import VehicleParams


class StanleyController:
    """
    Stanley lateral controller for path tracking.

    Control law:
        delta_cmd = theta_e + atan2(k * e_ct, v + v_stabilizer)
    where `theta_e` is heading error and `e_ct` is signed cross-track error.
    """

    def __init__(
        self,
        vehicle_params: VehicleParams,
        k: float = 0.5,
        v_stabilizer: float = 0.1,
    ):
        """
        Initialize the Stanley controller.

        Args:
            vehicle_params: Vehicle geometry/limits container.
            k: Cross-track gain.
            v_stabilizer: Small positive term to avoid division by zero at low speed.
        """
        self.vehicle_params = vehicle_params
        self.k = k
        self.v_stabilizer = v_stabilizer

    def _get_target_errors(self, state: np.ndarray, reference_path: ReferenceTrajectory) -> tuple[float, float]:
        """
        Compute cross-track and heading errors relative to nearest path point.

        Strategy:
            1. Project to front axle position.
            2. Find nearest reference sample to front axle.
            3. Compute heading error to path tangent at that sample.
            4. Compute signed cross-track error.

        Args:
            state: Vehicle state `[x, y, psi, v, delta]`.
            reference_path: Path samples with `x` and `y` coordinates.

        Returns:
            Tuple `(cross_track_error, heading_error)` in radians/meters.
        """
        # Stanley tracks using front axle position.
        pos = np.array([state[0], state[1]])  # rear axle position
        pos_front = pos + self.vehicle_params.wheelbase * np.array([np.cos(state[2]), np.sin(state[2])])
        path_points = np.column_stack([reference_path.x, reference_path.y])
        distances = np.linalg.norm(path_points - pos_front, axis=1)
        target_index = np.argmin(distances)
        path_point = path_points[target_index]
        cross_track_error_mag = distances[target_index]

        # Heading error between vehicle yaw and local path heading.
        theta_e = reference_path.psi[target_index] - state[2]
        theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))  # wrap to [-pi, pi]

        # Geometric sign from cross product:
        # sign( t_hat x e_hat ), where t_hat is path tangent and e_hat points
        # from path point to front axle.
        tangent = np.array([
            np.cos(reference_path.psi[target_index]),
            np.sin(reference_path.psi[target_index]),
        ])
        error_vec = pos_front - path_point
        cross_z = tangent[0] * error_vec[1] - tangent[1] * error_vec[0]
        cross_track_error = -np.sign(cross_z) * cross_track_error_mag
        return cross_track_error, theta_e

    def get_steering_command(self, state: np.ndarray, reference_path: ReferenceTrajectory) -> float:
        """
        Compute Stanley controller steering command.

        Args:
            state: Vehicle state `[x, y, psi, v, delta]`.
            reference_path: Desired path to track.

        Returns:
            Steering command `delta_cmd` [rad].
        """
        cross_track_error, theta_e = self._get_target_errors(state, reference_path)
        # Cross-track correction naturally weakens as speed increases.
        delta_command = theta_e + np.arctan2(self.k * cross_track_error, state[3] + self.v_stabilizer)
        # Enforce steering limits from vehicle model.
        delta_command = np.clip(delta_command, self.vehicle_params.delta_min, self.vehicle_params.delta_max)
        return delta_command
