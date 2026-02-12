import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, List


class SimulationPlotter:
    def __init__(self, dt: float, state_size: int):
        self.dt = dt
        self.state_size = state_size

    def _parse_state_indices(
        self,
        state_indices: Optional[Union[str, Sequence[int]]],
    ) -> List[int]:
        if state_indices is None:
            return list(range(self.state_size))

        if isinstance(state_indices, str):
            tokens = [token.strip() for token in state_indices.split(",") if token.strip()]
            indices = [int(token) for token in tokens]
        else:
            indices = [int(i) for i in state_indices]

        unique_indices = []
        for idx in indices:
            if idx < 0 or idx >= self.state_size:
                raise ValueError(f"state index {idx} out of range [0, {self.state_size - 1}]")
            if idx not in unique_indices:
                unique_indices.append(idx)

        if not unique_indices:
            raise ValueError("state_indices cannot be empty")
        return unique_indices

    def plot(
        self,
        states: np.ndarray,
        state_indices: Optional[Union[str, Sequence[int]]] = None,
        delta_cmd_seq: Optional[np.ndarray] = None,
        a_seq: Optional[np.ndarray] = None,
        show: bool = True,
    ):
        """
        Plot XY trajectory and selected state/input traces in one figure.
        """
        if states.ndim != 2 or states.shape[1] != self.state_size:
            raise ValueError(
                f"states must have shape (N, {self.state_size}), got {states.shape}"
            )

        selected_state_indices = self._parse_state_indices(state_indices)
        n_input_plots = int(delta_cmd_seq is not None) + int(a_seq is not None)
        total_plots = 1 + len(selected_state_indices) + n_input_plots

        fig, axes = plt.subplots(total_plots, 1, figsize=(10, 3 * total_plots), squeeze=False)
        axes = axes[:, 0]

        # Always plot x-y trajectory in the first subplot.
        ax_xy = axes[0]
        ax_xy.plot(states[:, 0], states[:, 1], label="trajectory")
        ax_xy.scatter(states[0, 0], states[0, 1], marker="o", label="start")
        ax_xy.scatter(states[-1, 0], states[-1, 1], marker="x", label="end")
        ax_xy.set_title("X-Y Trajectory")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.grid(True)
        ax_xy.axis("equal")
        ax_xy.legend()

        time_states = np.arange(states.shape[0]) * self.dt
        next_axis = 1

        for idx in selected_state_indices:
            ax = axes[next_axis]
            ax.plot(time_states, states[:, idx])
            ax.set_title(f"State {idx} vs Time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"state[{idx}]")
            ax.grid(True)
            next_axis += 1

        if delta_cmd_seq is not None:
            if len(delta_cmd_seq) == 0:
                raise ValueError("delta_cmd_seq cannot be empty when provided")
            ax = axes[next_axis]
            time_delta = np.arange(len(delta_cmd_seq)) * self.dt
            ax.step(time_delta, delta_cmd_seq, where="post")
            ax.set_title("Steering Command vs Time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("delta_cmd [rad]")
            ax.grid(True)
            next_axis += 1

        if a_seq is not None:
            if len(a_seq) == 0:
                raise ValueError("a_seq cannot be empty when provided")
            ax = axes[next_axis]
            time_a = np.arange(len(a_seq)) * self.dt
            ax.step(time_a, a_seq, where="post")
            ax.set_title("Acceleration vs Time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("a [m/s^2]")
            ax.grid(True)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
