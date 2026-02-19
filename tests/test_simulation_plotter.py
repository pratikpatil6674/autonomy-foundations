import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_plotter import SimulationPlotter


def make_states(offset: float = 0.0) -> np.ndarray:
    return np.array([
        [0.0 + offset, 0.0, 0.0, 5.0, 0.0],
        [1.0 + offset, 0.1, 0.0, 5.0, 0.0],
        [2.0 + offset, 0.2, 0.0, 5.0, 0.0],
        [3.0 + offset, 0.3, 0.0, 5.0, 0.0],
    ])


def test_plot_single_states_backward_compatible():
    plotter = SimulationPlotter(dt=0.1, state_size=5)
    states = make_states()

    fig, axes = plotter.plot(states=states, state_indices="0,1", show=False)

    assert len(axes) == 3  # xy + state0 + state1
    assert len(axes[0].lines) == 1
    assert len(axes[1].lines) == 1
    assert len(axes[2].lines) == 1
    plt.close(fig)


def test_plot_multiple_experiments_overlays_same_signal_type():
    plotter = SimulationPlotter(dt=0.1, state_size=5)
    states_list = [make_states(0.0), make_states(1.0), make_states(2.0)]

    fig, axes = plotter.plot(
        states=states_list,
        state_indices=[0, 1],
        experiment_labels=["true", "kf", "ekf"],
        show=False,
    )

    assert len(axes) == 3
    assert len(axes[0].lines) == 3  # three xy trajectories
    assert len(axes[1].lines) == 3  # state[0] for all experiments
    assert len(axes[2].lines) == 3  # state[1] for all experiments
    plt.close(fig)


def test_plot_reuses_single_input_sequence_across_experiments():
    plotter = SimulationPlotter(dt=0.1, state_size=5)
    states_list = [make_states(0.0), make_states(1.0)]
    delta_cmd_seq = np.array([0.0, 0.1, 0.2])

    fig, axes = plotter.plot(
        states=states_list,
        state_indices=[0],
        delta_cmd_seq=delta_cmd_seq,
        experiment_labels=["true", "kf"],
        show=False,
    )

    assert len(axes) == 3  # xy + state[0] + delta_cmd
    assert len(axes[2].lines) == 2
    plt.close(fig)


def test_plot_raises_on_label_length_mismatch():
    plotter = SimulationPlotter(dt=0.1, state_size=5)
    states_list = [make_states(0.0), make_states(1.0)]

    with pytest.raises(ValueError, match="experiment_labels must have length 2"):
        plotter.plot(
            states=states_list,
            state_indices=[0],
            experiment_labels=["only_one_label"],
            show=False,
        )
