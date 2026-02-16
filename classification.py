"""
classification.py – Evaluation tools and models for the EEG MI project
-------------------------------------------------------------
• Basic LDA with K-fold
• PCA + LDA
• RFE for feature selection
• Z-score normalization only
• Final test evaluation function
• Bonus: General classifier + Logistic Regression
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression          # ← Bonus
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from features import build_all_features
from visualization import np_load

# ─────────────────────────────────────────────────────────────
# 4.1  Basic LDA
# ─────────────────────────────────────────────────────────────
def evaluate_lda_crossval(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    random_state: int = 42,
):
    print("\n--- LDA Classification with k-fold Cross Validation ---")
    print(f"Data shape: X = {X.shape}, y = {y.shape}")
    print(f"Using k = {k} folds")

    lda = LinearDiscriminantAnalysis()
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    val_scores = cross_val_score(lda, X, y, cv=kf, scoring="accuracy")
    print(
        f"Validation Accuracy: {val_scores.mean()*100:.2f}% ± {val_scores.std()*100:.2f}%"
    )

    train_scores = []
    for train_idx, _ in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        lda.fit(X_train, y_train)
        y_train_pred = lda.predict(X_train)
        train_scores.append(accuracy_score(y_train, y_train_pred))

    train_scores = np.asarray(train_scores)
    print(
        f"Train Accuracy    : {train_scores.mean()*100:.2f}% ± {train_scores.std()*100:.2f}%"
    )
    return val_scores, train_scores


# ─────────────────────────────────────────────────────────────
# 4.2a  PCA + LDA
# ─────────────────────────────────────────────────────────────
def evaluate_with_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 3,
    k: int = 5,
):
    print(f"\n--- PCA-based Classification with top {n_components} components ---")
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=n_components).fit_transform(X_scaled)
    return evaluate_lda_crossval(X_pca, y, k=k)


# ─────────────────────────────────────────────────────────────
# 4.2b  RFE for feature selection
# ─────────────────────────────────────────────────────────────
def evaluate_with_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int | None = None,
    k: int = 5,
):
    print("\n--- RFE-based Feature Selection ---")
    if max_features is None:
        max_features = X.shape[1]

    scores: list[float] = []
    features = list(range(1, max_features + 1))
    best_selector: RFE | None = None

    for n_feat in features:
        lda = LinearDiscriminantAnalysis()
        selector = RFE(lda, n_features_to_select=n_feat)
        X_rfe = selector.fit_transform(X, y)
        val_scores = cross_val_score(
            lda, X_rfe, y, cv=KFold(n_splits=k, shuffle=True, random_state=42)
        )
        scores.append(val_scores.mean())
        if n_feat == np.argmax(scores) + 1:
            best_selector = selector

    plt.figure(figsize=(6, 4))
    plt.plot(features, scores, marker="o")
    plt.title("Validation Accuracy vs. Number of Selected Features (RFE)")
    plt.xlabel("Number of Features")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_n = np.argmax(scores) + 1
    print(
        f"Best number of features: {best_n} "
        f"with accuracy: {scores[best_n - 1] * 100:.2f}%"
    )

    if best_selector is not None:
        mask = best_selector.support_
        selected = [n for n, keep in zip(feature_names, mask) if keep]
        print("Selected features:", selected)

        selected_idx = [i for i, n in enumerate(feature_names) if n in selected]
        X_best = X[:, selected_idx]
        print(f"\n[4.2c] LDA with only top {len(selected)} selected features:")
        evaluate_lda_crossval(X_best, y, k=k)

        if len(selected_idx) >= 3:
            lda_full = LinearDiscriminantAnalysis()
            rfe4 = RFE(lda_full, n_features_to_select=4)
            X_rfe4 = rfe4.fit_transform(X, y)
            print("\n[4.2d] LDA with top 4 features (for comparison):")
            evaluate_lda_crossval(X_rfe4, y, k=k)

    return best_n, scores


# ─────────────────────────────────────────────────────────────
# 4.2e  Z-score normalization only
# ─────────────────────────────────────────────────────────────
def evaluate_on_normalized_features(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
):
    print("\n[4.2e] LDA on Z-score Normalized Features (no PCA or RFE):")
    X_scaled = StandardScaler().fit_transform(X)
    evaluate_lda_crossval(X_scaled, y, k=k)


# ─────────────────────────────────────────────────────────────
#  Final classifier + test evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_final_classifier_on_test(
    train_data,
    train_labels,
    test_data,
    test_labels,
    fs,
    feature_names,
):
    # ➊  Full feature extraction – Train
    X_train, y_train, _ = build_all_features(train_data, train_labels, fs)

    # ➋  RFE for selecting 3 features
    lda = LinearDiscriminantAnalysis()
    rfe = RFE(lda, n_features_to_select=3).fit(X_train, y_train)
    X_train_sel = rfe.transform(X_train)

    # ➌  Z-score normalization
    scaler = StandardScaler().fit(X_train_sel)
    X_train_scaled = scaler.transform(X_train_sel)

    # ➍  Final LDA training
    clf = LinearDiscriminantAnalysis().fit(X_train_scaled, y_train)

    # ➎  Test feature extraction
    X_test, y_test, _ = build_all_features(test_data, test_labels, fs)
    X_test_sel = rfe.transform(X_test)
    X_test_scaled = scaler.transform(X_test_sel)

    # ➏  Prediction + accuracy
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    np.save("test_predictions.npy", y_pred)
    print("Saved predictions to test_predictions.npy")


# ============================================================================#
# =============================   BONUS CODE   ===============================#
# ============================================================================#

# ─────────────────────────────────────────────────────────────
#  B1  –  Generic classifier evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_classifier_crossval(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    k: int = 5,
    random_state: int = 42,
    name: str | None = None,
):
    """
    Generic K-fold evaluation tool for *any* sklearn classifier.

    Parameters
    ----------
    clf  : sklearn.base.BaseEstimator
        A classifier object (e.g., LogisticRegression()).
    name : str
        Name to print; default is the class name.
    """
    name = name or clf.__class__.__name__
    print(f"\n--- {name} with {k}-fold Cross Validation ---")
    print(f"Data shape: X = {X.shape}, y = {y.shape}")

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Validation
    val_scores = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")
    print(
        f"Validation Accuracy: {val_scores.mean()*100:.2f}% ± "
        f"{val_scores.std()*100:.2f}%"
    )

    # Training
    train_scores = []
    for train_idx, _ in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        clf.fit(X_tr, y_tr)
        train_scores.append(accuracy_score(y_tr, clf.predict(X_tr)))
    train_scores = np.asarray(train_scores)
    print(
        f"Train Accuracy    : {train_scores.mean()*100:.2f}% ± "
        f"{train_scores.std()*100:.2f}%"
    )
    return val_scores, train_scores


# ────────────────────────────────────────────────────────────
#  B2  –  Logistic Regression specific 
# ─────────────────────────────────────────────────────────────
def evaluate_logreg_crossval(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
):
    """K-fold CV for Logistic Regression (linear)."""
    logreg = LogisticRegression(
        penalty="l2",
        solver="liblinear",   # Suitable for small datasets
        max_iter=1000,
    )
    return evaluate_classifier_crossval(X, y, logreg, k)
