import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Sequence, Union, List, Tuple


class SimulationPlotter:
    """
    Utility class for visualizing vehicle simulation outputs.

    The first subplot always shows the X-Y trajectory. Additional subplots
    show selected state traces and optional control-input traces.
    """

    def __init__(self, dt: float, state_size: int):
        """
        Initialize the plotter.

        Args:
            dt: Sampling time in seconds.
            state_size: Number of state variables in each state vector.
        """
        self.dt = dt
        self.state_size = state_size
        self._last_animation: Optional[FuncAnimation] = None

    def _parse_state_indices(
        self,
        state_indices: Optional[Union[str, Sequence[int]]],
    ) -> List[int]:
        """
        Parse and validate selected state indices.

        Args:
            state_indices:
                - `None`: plot all state indices `[0, ..., state_size - 1]`.
                - `str`: comma-separated values, e.g. `"0,2,4"`.
                - `Sequence[int]`: explicit list/tuple/array of indices.

        Returns:
            Ordered list of unique, validated indices.

        Raises:
            ValueError: If any index is out of range or the parsed selection is empty.
        """
        if state_indices is None:
            return list(range(self.state_size))

        if isinstance(state_indices, str):
            # Allow whitespace in comma-separated input like "0, 2, 4".
            tokens = [token.strip() for token in state_indices.split(",") if token.strip()]
            indices = [int(token) for token in tokens]
        else:
            indices = [int(i) for i in state_indices]

        unique_indices = []
        for idx in indices:
            if idx < 0 or idx >= self.state_size:
                raise ValueError(f"state index {idx} out of range [0, {self.state_size - 1}]")
            # Preserve first-seen order while removing duplicates.
            if idx not in unique_indices:
                unique_indices.append(idx)

        if not unique_indices:
            raise ValueError("state_indices cannot be empty")
        return unique_indices

    def _normalize_states_input(
        self,
        states: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Normalize state input to a list of state arrays.

        Each state array must have shape `(N, state_size)`.
        """
        if isinstance(states, np.ndarray):
            states_list = [states]
        else:
            states_list = [np.asarray(s) for s in states]
            if len(states_list) == 0:
                raise ValueError("states sequence cannot be empty")

        for i, state_arr in enumerate(states_list):
            if state_arr.ndim != 2 or state_arr.shape[1] != self.state_size:
                raise ValueError(
                    f"states[{i}] must have shape (N, {self.state_size}), got {state_arr.shape}"
                )
        return states_list

    def _normalize_input_series(
        self,
        series: Optional[Union[np.ndarray, Sequence[np.ndarray]]],
        name: str,
        n_experiments: int,
    ) -> Optional[List[np.ndarray]]:
        """
        Normalize optional input series to one 1D array per experiment.

        Accepts:
            - None
            - Single array (shared across all experiments)
            - Sequence of arrays with length 1 or `n_experiments`
        """
        if series is None:
            return None

        if isinstance(series, np.ndarray):
            if series.ndim != 1:
                raise ValueError(f"{name} must be a 1D array, got shape {series.shape}")
            if len(series) == 0:
                raise ValueError(f"{name} cannot be empty when provided")
            return [series] * n_experiments

        series_list = [np.asarray(s) for s in series]
        if len(series_list) == 0:
            raise ValueError(f"{name} sequence cannot be empty when provided")
        if len(series_list) not in (1, n_experiments):
            raise ValueError(
                f"{name} sequence must have length 1 or {n_experiments}, got {len(series_list)}"
            )

        for i, seq in enumerate(series_list):
            if seq.ndim != 1:
                raise ValueError(f"{name}[{i}] must be a 1D array, got shape {seq.shape}")
            if len(seq) == 0:
                raise ValueError(f"{name}[{i}] cannot be empty when provided")

        if len(series_list) == 1:
            return [series_list[0]] * n_experiments
        return series_list

    def _make_experiment_labels(
        self,
        n_experiments: int,
        experiment_labels: Optional[Sequence[str]],
    ) -> List[str]:
        """Build experiment labels and validate their count."""
        if experiment_labels is None:
            if n_experiments == 1:
                return ["trajectory"]
            return [f"experiment_{i}" for i in range(n_experiments)]

        labels = [str(label) for label in experiment_labels]
        if len(labels) != n_experiments:
            raise ValueError(
                f"experiment_labels must have length {n_experiments}, got {len(labels)}"
            )
        return labels

    def plot(
        self,
        states: Union[np.ndarray, Sequence[np.ndarray]],
        state_indices: Optional[Union[str, Sequence[int]]] = None,
        delta_cmd_seq: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        a_seq: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        experiment_labels: Optional[Sequence[str]] = None,
        animate: bool = False,
        animate_interval_ms: int = 30,
        animate_stride: int = 1,
        animate_heading_length: float = 0.8,
        show: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot trajectory, selected state traces, and optional input traces.

        Args:
            states:
                Single state array `(N, state_size)` or sequence of state arrays
                from different experiments.
            state_indices:
                Which state dimensions to plot as time series. Accepts `None`,
                comma-separated string, or sequence of indices.
            delta_cmd_seq:
                Optional steering-command series. Can be a single sequence (shared
                across all experiments) or one sequence per experiment.
            a_seq:
                Optional acceleration series. Can be a single sequence (shared
                across all experiments) or one sequence per experiment.
            experiment_labels:
                Optional labels for experiments. Length must match number of
                experiments in `states`.
            animate:
                If `True`, animate the X-Y subplot to show motion over time.
            animate_interval_ms:
                Delay between animation frames in milliseconds.
            animate_stride:
                Use every `animate_stride` sample to speed up animation.
            animate_heading_length:
                Arrow length used to indicate heading direction from `psi`.
            show: If `True`, call `matplotlib.pyplot.show()`.

        Returns:
            Tuple `(fig, axes)` from Matplotlib.
                - `fig`: created `matplotlib.figure.Figure`.
                - `axes`: 1D array of subplot axes in draw order.

        Raises:
            ValueError: If state/input shapes are invalid or label lengths mismatch.
        """
        states_list = self._normalize_states_input(states)
        n_experiments = len(states_list)
        labels = self._make_experiment_labels(n_experiments, experiment_labels)
        delta_cmd_list = self._normalize_input_series(delta_cmd_seq, "delta_cmd_seq", n_experiments)
        a_list = self._normalize_input_series(a_seq, "a_seq", n_experiments)

        selected_state_indices = self._parse_state_indices(state_indices)
        n_input_plots = int(delta_cmd_list is not None) + int(a_list is not None)
        total_plots = 1 + len(selected_state_indices) + n_input_plots

        fig, axes = plt.subplots(total_plots, 1, figsize=(10, 3 * total_plots), squeeze=False)
        axes = axes[:, 0]

        # Plot x-y trajectories from all experiments in the first subplot.
        ax_xy = axes[0]
        for exp_idx, state_arr in enumerate(states_list):
            line = ax_xy.plot(state_arr[:, 0], state_arr[:, 1], label=labels[exp_idx])[0]
            color = line.get_color()
            ax_xy.scatter(state_arr[0, 0], state_arr[0, 1], marker="o", color=color)
            ax_xy.scatter(state_arr[-1, 0], state_arr[-1, 1], marker="x", color=color)
        ax_xy.set_title("X-Y Trajectory")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.grid(True)
        ax_xy.axis("equal")
        ax_xy.legend()

        if animate:
            if animate_stride < 1:
                raise ValueError(f"animate_stride must be >= 1, got {animate_stride}")
            if animate_interval_ms <= 0:
                raise ValueError(f"animate_interval_ms must be > 0, got {animate_interval_ms}")
            if animate_heading_length <= 0.0:
                raise ValueError(
                    f"animate_heading_length must be > 0, got {animate_heading_length}"
                )

            trail_lines = []
            car_arrows = []
            frame_count = max(1, max(state_arr.shape[0] for state_arr in states_list) // animate_stride)

            for exp_idx, state_arr in enumerate(states_list):
                if state_arr.shape[1] <= 2:
                    raise ValueError(
                        "animate=True requires state arrays with psi at index 2."
                    )
                line = ax_xy.plot([], [], linestyle="-", linewidth=2.0, label=f"{labels[exp_idx]} (trail)")[0]
                x0, y0, psi0 = state_arr[0, 0], state_arr[0, 1], state_arr[0, 2]
                arrow = ax_xy.quiver(
                    [x0],
                    [y0],
                    [animate_heading_length * np.cos(psi0)],
                    [animate_heading_length * np.sin(psi0)],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=line.get_color(),
                )
                trail_lines.append((line, state_arr))
                car_arrows.append((arrow, state_arr))

            # Refresh legend to include animated artists.
            ax_xy.legend()

            def _init_anim():
                for line, _ in trail_lines:
                    line.set_data([], [])
                for arrow, state_arr in car_arrows:
                    x0, y0, psi0 = state_arr[0, 0], state_arr[0, 1], state_arr[0, 2]
                    arrow.set_offsets(np.array([[x0, y0]]))
                    arrow.set_UVC(
                        animate_heading_length * np.cos(psi0),
                        animate_heading_length * np.sin(psi0),
                    )
                return [a for a, _ in trail_lines] + [a for a, _ in car_arrows]

            def _update_anim(frame_idx: int):
                sample_idx = frame_idx * animate_stride
                artists = []
                for line, state_arr in trail_lines:
                    idx = min(sample_idx, state_arr.shape[0] - 1)
                    line.set_data(state_arr[: idx + 1, 0], state_arr[: idx + 1, 1])
                    artists.append(line)
                for arrow, state_arr in car_arrows:
                    idx = min(sample_idx, state_arr.shape[0] - 1)
                    x, y, psi = state_arr[idx, 0], state_arr[idx, 1], state_arr[idx, 2]
                    arrow.set_offsets(np.array([[x, y]]))
                    arrow.set_UVC(
                        animate_heading_length * np.cos(psi),
                        animate_heading_length * np.sin(psi),
                    )
                    artists.append(arrow)
                return artists

            self._last_animation = FuncAnimation(
                fig=fig,
                func=_update_anim,
                init_func=_init_anim,
                frames=frame_count,
                interval=animate_interval_ms,
                blit=True,
                repeat=False,
            )

        next_axis = 1

        for idx in selected_state_indices:
            ax = axes[next_axis]
            for exp_idx, state_arr in enumerate(states_list):
                # State timeline includes the initial state.
                time_states = np.arange(state_arr.shape[0]) * self.dt
                ax.plot(time_states, state_arr[:, idx], label=labels[exp_idx])
            ax.set_title(f"State {idx} vs Time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"state[{idx}]")
            ax.grid(True)
            ax.legend()
            next_axis += 1

        if delta_cmd_list is not None:
            ax = axes[next_axis]
            # Inputs are displayed as piecewise-constant commands.
            for exp_idx, delta_seq in enumerate(delta_cmd_list):
                time_delta = np.arange(len(delta_seq)) * self.dt
                ax.step(time_delta, delta_seq, where="post", label=labels[exp_idx])
            ax.set_title("Steering Command vs Time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("delta_cmd [rad]")
            ax.grid(True)
            ax.legend()
            next_axis += 1

        if a_list is not None:
            ax = axes[next_axis]
            # Inputs are displayed as piecewise-constant commands.
            for exp_idx, accel_seq in enumerate(a_list):
                time_a = np.arange(len(accel_seq)) * self.dt
                ax.step(time_a, accel_seq, where="post", label=labels[exp_idx])
            ax.set_title("Acceleration vs Time")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("a [m/s^2]")
            ax.grid(True)
            ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
