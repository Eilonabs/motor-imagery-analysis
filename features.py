import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

# ============================================================
# 1. Band‑power features (3 bands)
# ============================================================

BAND_DEFS = [
    (4.0, 6.0, 0, 12, 20, "12‑20Hz_C3"),
    (4.0, 6.0, 0, 30, 35, "30‑35Hz_C3"),
    (1.8, 4.1, 1, 15, 22, "15‑22Hz_C4"),
]


def _band_power(sig: np.ndarray, fs: int, f_low: float, f_high: float) -> float:
    f, Pxx = welch(sig, fs=fs, nperseg=fs)
    band = Pxx[(f >= f_low) & (f <= f_high)].sum()
    return 10 * np.log10(band + 1e-12)

# ============================================================
# Helper – assure labels are (n_trials,4)
# ============================================================

def _ensure_trial_first(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2 or 4 not in labels.shape:
        raise ValueError("labels must be 2-D and include 4 columns (ARTIFACT, REMOVE, LEFT, RIGHT)")
    return labels.T if labels.shape[0] == 4 else labels

# ============================================================
# 2. Band‑power feature extraction
# ============================================================

def extract_band_power_features(data: np.ndarray, labels: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = _ensure_trial_first(labels)
    sel_idx = (labels[:, 2] == 1) | (labels[:, 3] == 1)
    y = np.where(labels[sel_idx, 3] == 1, 1, 0)

    X = []
    for trial in data[sel_idx]:
        feats = []
        for t0, t1, ch, f_low, f_high, _ in BAND_DEFS:
            s0, s1 = int(t0 * fs), int(t1 * fs)
            sig = trial[s0:s1, ch] - trial[s0:s1, ch].mean()
            feats.append(_band_power(sig, fs, f_low, f_high))
        X.append(feats)

    feature_names = [b[-1] for b in BAND_DEFS]
    return np.array(X), y, feature_names

# ============================================================
# 3. Zero‑crossing rate + log variance ratio
# ============================================================

def zero_cross_rate(sig: np.ndarray) -> float:
    """Zero‑crossing rate per sample (fraction)."""
    crossings = np.where(np.diff(np.signbit(sig)))[0]
    return len(crossings) / (len(sig) - 1 + 1e-12)


def extract_additional_features(data: np.ndarray, labels: np.ndarray, fs: int) -> Tuple[np.ndarray, List[str]]:
    labels = _ensure_trial_first(labels)
    sel_idx = (labels[:, 2] == 1) | (labels[:, 3] == 1)

    feats_all = []
    for trial in data[sel_idx]:
        s0, s1 = int(4.0 * fs), int(6.0 * fs)
        sig_c3 = trial[s0:s1, 0] - trial[s0:s1, 0].mean()
        sig_c4 = trial[s0:s1, 1] - trial[s0:s1, 1].mean()

        # Feature A: Zero‑crossing rate in C3
        zcr_c3 = zero_cross_rate(sig_c3)

        # Feature B: log variance ratio C3/C4
        var_ratio = np.var(sig_c3) / (np.var(sig_c4) + 1e-12)
        feats_all.append([zcr_c3, np.log(var_ratio + 1e-12)])

    return np.array(feats_all), ["ZCR_C3", "logVarRatio_C3/C4"]

# ============================================================
# 4. Histogram plotting
# ============================================================

def plot_feature_histograms(X: np.ndarray, y: np.ndarray, names: List[str]):
    for idx, name in enumerate(names):
        xmin, xmax = X[:, idx].min(), X[:, idx].max()
        plt.figure(figsize=(6, 4))
        sns.histplot(X[y == 0, idx], color="tab:blue", alpha=0.6,
                     label=f"Left  (n={np.sum(y==0)})", kde=True, bins='auto')
        sns.histplot(X[y == 1, idx], color="tab:red", alpha=0.6,
                     label=f"Right (n={np.sum(y==1)})", kde=True, bins='auto')
        plt.xlim(xmin, xmax)
        plt.title(f"Histogram of {name}")
        plt.xlabel("Value"); plt.ylabel("Count")
        plt.legend(); plt.tight_layout(); plt.show()

# ============================================================
# 5. PCA visualisation
# ============================================================

def pca_visualization(X: np.ndarray, y: np.ndarray, scaled: bool = False):
    X_proc = StandardScaler().fit_transform(X) if scaled else X
    comps  = PCA(n_components=3).fit_transform(X_proc)
    # 2D plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=y, palette="Set1", alpha=0.8)
    plt.title(f"PCA 2D (scaled={scaled})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
    # 3D plot with legend
    from mpl_toolkits.mplot3d import Axes3D  
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        indices = y == label
        ax.scatter(comps[indices, 0], comps[indices, 1], comps[indices, 2],
                   color=colors[i], label=str(label), alpha=0.8)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"PCA 3D (scaled={scaled})")
    ax.legend(title="Label")
    fig.tight_layout()
    plt.show()
# ============================================================
# 6. Master helper
# ============================================================

def build_all_features(data: np.ndarray, labels: np.ndarray, fs: int):
    labels   = _ensure_trial_first(labels)
    X_band, y, names_band = extract_band_power_features(data, labels, fs)
    X_extra, names_extra  = extract_additional_features(data, labels, fs)
    X_total               = np.hstack([X_band, X_extra])
    return X_total, y, names_band + names_extra
