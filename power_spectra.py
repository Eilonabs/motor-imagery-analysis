import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from typing import Tuple

# ------------------------------------------------------------------
# 1. Power Spectrum (Welch) with log10 stability
# ------------------------------------------------------------------
def extract_trials(data: np.ndarray, labels: np.ndarray,
                   fs: int, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extract left/right trials using only samples between start and end (in seconds)."""
    left = labels[2] == 1
    right = labels[3] == 1
    t0, t1 = int(start * fs), int(end * fs)
    return data[left, t0:t1, :], data[right, t0:t1, :]

def compute_avg_psd_db(trials: np.ndarray, fs: int):
    """Compute Welch-PSD in dB scale (log10), with mean & std across trials."""
    psd_all_db = []
    for trial in trials:
        per_trial_db = []
        for ch in range(trial.shape[1]):
            sig = trial[:, ch] - np.mean(trial[:, ch])
            nseg = min(fs, len(sig))
            f, pxx = welch(sig, fs=fs, nperseg=nseg, detrend='constant')
            pxx_db = 10 * np.log10(pxx + 1e-12)
            per_trial_db.append(pxx_db)
        psd_all_db.append(np.stack(per_trial_db, axis=1))
    psd_all_db = np.stack(psd_all_db)
    return f, psd_all_db.mean(axis=0), psd_all_db.std(axis=0)

def plot_avg_psd_db(f, l_db, l_std, r_db, r_std):
    """Plot Welch power spectrum in dB with ±1 std for Left/Right × C3/C4."""
    keep = (f >= 1) & (f <= 40)
    f = f[keep]
    l_db, l_std = l_db[keep], l_std[keep]
    r_db, r_std = r_db[keep], r_std[keep]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ch, ax in enumerate(axs):
        ax.plot(f, l_db[:, ch], 'b', label='Left')
        # === Bonus: added ±1 standard deviation shading for Left-hand imagery ===
        ax.fill_between(f, l_db[:, ch] - l_std[:, ch], l_db[:, ch] + l_std[:, ch],
                        color='b', alpha=0.25)

        ax.plot(f, r_db[:, ch], 'r', label='Right')
        # === Bonus: added ±1 standard deviation shading for Right-hand imagery ===
        ax.fill_between(f, r_db[:, ch] - r_std[:, ch], r_db[:, ch] + r_std[:, ch],
                        color='r', alpha=0.25)

        ax.set_title(f"Channel {'C3' if ch == 0 else 'C4'}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.legend()
    plt.suptitle("Welch Power Spectrum (4–6 s, 1–40 Hz)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# 2. Spectrograms – whole trial, +baseline correction
# ------------------------------------------------------------------
def average_spectrogram(data: np.ndarray, fs: int, chan: int):
    specs = []
    for trial in data:
        f, t, Sxx = spectrogram(trial[:, chan] - trial[:, chan].mean(),
                                fs=fs, nperseg=fs // 2)
        specs.append(Sxx)
    return f, t, np.mean(specs, axis=0)

def baseline_corrected(data: np.ndarray, fs: int, chan: int):
    all_specs = []
    for trial in data:
        f, t, Sxx = spectrogram(trial[:, chan] - trial[:, chan].mean(),
                                fs=fs, nperseg=fs // 2)
        baseline = Sxx[:, t < 1.0].mean(axis=1, keepdims=True)
        all_specs.append(10 * np.log10(Sxx / (baseline + 1e-10)))
    return np.mean(all_specs, axis=0), f, t

def _plot_spec(ax, spec, f, t, title):
    im = ax.imshow(spec, origin="lower", aspect="auto",
                   extent=[t[0], t[-1], f[0], f[-1]], cmap="seismic")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax, label="dB")

# ------------------------------------------------------------------
# 3. Master wrapper function
# ------------------------------------------------------------------
def full_power_analysis(data, labels, fs):
    # ----------- Power Spectrum (in dB) -----------
    left, right = extract_trials(data, labels, fs, 4, 6)  # <––– window adjusted to 4–6s
    f, l_db, l_std = compute_avg_psd_db(left, fs)
    _, r_db, r_std = compute_avg_psd_db(right, fs)
    plot_avg_psd_db(f, l_db, l_std, r_db, r_std)

    # ----------- Spectrogram Ratio -----------
    left_mask, right_mask = labels[2] == 1, labels[3] == 1

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    f, t, L_c3 = average_spectrogram(data[left_mask], fs, chan=0)
    _, _, R_c3 = average_spectrogram(data[right_mask], fs, chan=0)
    _plot_spec(axs[0], 10 * np.log10(R_c3 / (L_c3 + 1e-10)), f, t, "Right / Left (C3)")

    f, t, L_c4 = average_spectrogram(data[left_mask], fs, chan=1)
    _, _, R_c4 = average_spectrogram(data[right_mask], fs, chan=1)
    _plot_spec(axs[1], 10 * np.log10(R_c4 / (L_c4 + 1e-10)), f, t, "Right / Left (C4)")
    plt.suptitle("Spectrogram Ratio (full trial)")
    plt.tight_layout()
    plt.show()

    # ----------- Baseline-corrected Spectrogram Difference -----------
    L_bc_c3, f, t = baseline_corrected(data[left_mask], fs, chan=0)
    R_bc_c3, _, _ = baseline_corrected(data[right_mask], fs, chan=0)
    L_bc_c4, _, _ = baseline_corrected(data[left_mask], fs, chan=1)
    R_bc_c4, _, _ = baseline_corrected(data[right_mask], fs, chan=1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    _plot_spec(axs[0], R_bc_c3 - L_bc_c3, f, t, "Baseline Δ (C3)")
    _plot_spec(axs[1], R_bc_c4 - L_bc_c4, f, t, "Baseline Δ (C4)")
    plt.suptitle("Baseline-Corrected Spectrogram Difference")
    plt.tight_layout()
    plt.show()
