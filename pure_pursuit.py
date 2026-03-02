import numpy as np

from params import VehicleParams
from reference_trajectories import ReferenceTrajectory


class PurePursuit:
    """
    Pure pursuit lateral controller for kinematic bicycle-style tracking.

    The controller computes a lookahead point on the reference path and
    returns a steering command that aims the vehicle toward that point.
    """

    def __init__(
        self,
        vehicle_params: VehicleParams,
        Lmin: float = 1.0,
        k: float = 0.5
    ):
        """
        Initialize the pure pursuit controller.

        Args:
            vehicle_params: Vehicle geometry/limits container.
            Lmin: Minimum lookahead distance [m].
            k: Speed gain in lookahead law, Ld = Lmin + k * v.
        """
        self.vehicle_params = vehicle_params
        self.Lmin = Lmin
        self.k = k

    def _compute_lookahead_distance(self, state: np.ndarray) -> float:
        """
        Compute lookahead distance from current vehicle speed.

        Args:
            state: Vehicle state `[x, y, psi, v, delta]`.

        Returns:
            Lookahead distance `Ld` in meters.
        """
        speed = state[3]
        Ld = self.Lmin + self.k * speed
        return Ld

    def _get_alpha(self, state: np.ndarray, target_position: np.ndarray) -> float:
        """
        Compute heading error to the target point in vehicle coordinates.

        Args:
            state: Vehicle state `[x, y, psi, v, delta]`.
            target_position: Target point `[x_t, y_t]` on the reference path.

        Returns:
            Bearing error `alpha` [rad].
        """
        alpha = np.arctan2(target_position[1] - state[1], target_position[0] - state[0]) - state[2]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # wrap to [-pi, pi]
        return alpha

    def _get_target_position(self, state: np.ndarray, reference_path: ReferenceTrajectory) -> np.ndarray:
        """
        Select the lookahead target point along the reference path.

        Strategy:
            1. Find nearest path sample to current position.
            2. March forward along the path until traveled arc length >= Ld.

        Args:
            state: Vehicle state `[x, y, psi, v, delta]`.
            reference_path: Path samples with `x` and `y` coordinates.

        Returns:
            Target point `[x_t, y_t]`.
        TODO : Optimize using last nearest point index
        """
        pos = np.array([state[0], state[1]])
        Ld = self._compute_lookahead_distance(state)
        path_points = np.column_stack([reference_path.x, reference_path.y])
        distances = np.linalg.norm(path_points - pos, axis=1)
        target_index = np.argmin(distances)
        dist = 0
        # Accumulate path arc length from nearest point to the lookahead point.
        while dist < Ld and target_index < len(path_points) - 1:
            dist += np.linalg.norm(path_points[target_index] - path_points[target_index + 1])
            target_index += 1
        return path_points[target_index]

    def get_steering_command(self, state: np.ndarray, reference_path: ReferenceTrajectory) -> float:
        """
        Compute pure pursuit steering command.

        Args:
            state: Vehicle state `[x, y, psi, v, delta]`.
            reference_path: Desired path to track.

        Returns:
            Steering command `delta_cmd` [rad].
        """
        target_position = self._get_target_position(state, reference_path)
        alpha = self._get_alpha(state, target_position)
        # Classic pure pursuit law for bicycle geometry.
        delta = np.arctan(2 * self.vehicle_params.wheelbase * np.sin(alpha) / self._compute_lookahead_distance(state))
        delta = np.clip(delta, self.vehicle_params.delta_min, self.vehicle_params.delta_max)
        return delta
