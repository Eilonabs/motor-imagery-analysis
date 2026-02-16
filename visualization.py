import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

np_load = lambda f: np.load(f, allow_pickle=True)

def extract_valid_trials(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    No artifacts: each trial belongs exclusively to LEFT or RIGHT.
    Returns:
      X –  (n_trials, n_samples, 2)  ← channels C3,C4
      y –  (n_trials)                ← 0=LEFT, 1=RIGHT
    """
    left_mask  = labels[2] == 1
    right_mask = labels[3] == 1
    mask = left_mask | right_mask       # All relevant trials

    X = data[mask, :, :2]               # First two channels
    y = np.zeros(mask.sum(), dtype=int)
    y[right_mask[mask]] = 1             # RIGHT → 1
    return X, y


def plot_random_trials(X: np.ndarray, y: np.ndarray, label_value: int, title: str) -> None:
    """Displays 20 random trials from the requested class (Left/Right)."""
    np.random.seed(5)
    idx = np.where(y == label_value)[0]
    chosen = np.random.choice(idx, 20, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        trial = X[chosen[i]]
        ax.plot(trial[:, 0], label="C3", color="blue", linewidth=0.7)
        ax.plot(trial[:, 1], label="C4", color="red",  linewidth=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Trial {chosen[i]}")
    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.04, "Time (samples)",              ha='center')
    fig.text(0.04, 0.5, "Amplitude (µV)", va='center', rotation='vertical')
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right')
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.94])
    plt.show()
