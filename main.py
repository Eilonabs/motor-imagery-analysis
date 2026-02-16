"""
main.py – Full project pipeline
--------------------------------
1. Basic trial visualization
2. Power Spectra + Spectrograms
3. Feature extraction, histograms and PCA
4. LDA classification and accuracy improvement
   + Bonus: Logistic-Regression classifier
5. Final test-set evaluation  |  prediction-only if no GT labels
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

# ── Project modules ───────────────────────────────────────────────────────────
from visualization import np_load, extract_valid_trials, plot_random_trials
from power_spectra import full_power_analysis
from features import (
    build_all_features,
    plot_feature_histograms,
    pca_visualization,
)
from classification import (
    evaluate_lda_crossval,
    evaluate_with_pca,
    evaluate_with_rfe,
    evaluate_on_normalized_features,
    evaluate_final_classifier_on_test,
    evaluate_logreg_crossval,        # ← Bonus import (לוגיסטי)
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# ── Utility ───────────────────────────────────────────────────────────────────
def _load_npy_sequence(path: str | Path) -> list[np.ndarray]:
    """Return *all* numpy arrays stored consecutively in a single .npy file."""
    out: list[np.ndarray] = []
    with open(path, "rb") as f:
        while True:
            try:
                out.append(np_load(f))
            except EOFError:
                break
    return out


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    train_path = Path("motor_imagery_train_data.npy")  
    test_path  = Path("motor_imagery_test_data.npy")   

    # === 0. Load TRAIN ========================================================
    trn = _load_npy_sequence(train_path)
    if len(trn) != 4:
        raise ValueError(
            f"{train_path} must contain 4 arrays "
            f"(data, labels, labels_name, fs) – found {len(trn)}."
        )
    train_data, labels, labels_name, fs_arr = trn
    fs = int(fs_arr)

    # === 1. Visualize random trials ==========================================
    X_vis, y_vis = extract_valid_trials(train_data, labels)
    plot_random_trials(X_vis, y_vis, 0, "Left-hand imagery – 20 trials")
    plot_random_trials(X_vis, y_vis, 1, "Right-hand imagery – 20 trials")

    # === 2. Power Spectra + Spectrograms =====================================
    full_power_analysis(train_data, labels, fs)

    # === 3. Feature extraction, histograms & PCA =============================
    X_feat, y_feat, feature_names = build_all_features(train_data, labels, fs)
    print("Feature matrix shape:", X_feat.shape)
    print("Labels shape:",    y_feat.shape)

    plot_feature_histograms(X_feat, y_feat, feature_names)
    pca_visualization(X_feat, y_feat, scaled=False)
    pca_visualization(X_feat, y_feat, scaled=True)

    # === 4. Classification experiments =======================================
    print("\n[4.1] LDA Classification with K-Fold Cross-Validation:")
    evaluate_lda_crossval(X_feat, y_feat, k=5)

    # ----- Bonus classifier ---------------------------------------------------
    print("\n[4.1-Bonus] Logistic Regression Classifier:")
    evaluate_logreg_crossval(X_feat, y_feat, k=5)

    print("\n[4.2a] LDA with PCA (2 components):")
    evaluate_with_pca(X_feat, y_feat, n_components=2)


    print("\n[4.2b] LDA with RFE – Feature Ranking Selection:")
    evaluate_with_rfe(
        X_feat, y_feat,
        feature_names=feature_names,
        max_features=X_feat.shape[1],
    )

    print("\n[4.2e] LDA on Z-score Normalized Features:")
    evaluate_on_normalized_features(X_feat, y_feat)

    # === 5. Final evaluation / prediction on TEST ============================
    print("\n[5.0] Final evaluation on test data:")

    tst = _load_npy_sequence(test_path)
    if len(tst) == 4:                     # data + GT labels exist
        test_data, test_labels, test_labels_name, _ = tst
        evaluate_final_classifier_on_test(
            train_data, labels, test_data, test_labels, fs, feature_names
        )

    elif len(tst) == 1:                   # only data → prediction-only mode
        test_data = tst[0]
        print(
            "⚠️  No ground-truth labels found – "
            "running prediction-only and saving to 'test_predictions.npy'."
        )

        # --- Train final classifier on FULL train set (3 best RFE features) ---
        X_train, y_train, _ = build_all_features(train_data, labels, fs)
        lda_base = LinearDiscriminantAnalysis()
        rfe = RFE(lda_base, n_features_to_select=3).fit(X_train, y_train)
        scaler = StandardScaler().fit(rfe.transform(X_train))
        clf = LinearDiscriminantAnalysis().fit(
            scaler.transform(rfe.transform(X_train)), y_train
        )

        # ---  mark each trial as LEFT to avoid filtering ----------
        dummy_labels = np.zeros((len(test_data), 4), dtype=int)
        dummy_labels[:, 2] = 1            # LEFT column = 1
        X_test, _, _ = build_all_features(test_data, dummy_labels, fs)

        # --- Predict & save ----------------------------------------------------
        y_pred = clf.predict(scaler.transform(rfe.transform(X_test)))
        np.save("test_predictions.npy", y_pred)
        print("Predictions saved ➜ test_predictions.npy")

    else:
        raise ValueError(
            f"{test_path} contains {len(tst)} arrays – expected 1 or 4."
        )


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
