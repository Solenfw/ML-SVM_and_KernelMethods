"""Platt Scaling (Lin, Lin & Weng, 2007) and Pairwise Coupling (Wu, Lin & Weng, 2004).

Implements:
- Numerically stable Newton's method with backtracking line search to fit
  P(y = +1 | f) = 1 / (1 + exp(A * f + B)) minimizing regularized negative log-likelihood.
- Internal stratified K-fold cross-validation so decision values f are out-of-fold.
- Multiclass pairwise coupling (Algorithm 2 of Wu, Lin & Weng 2004) for OvO probabilities.
- Calibration metrics: Brier score and reliability diagram binning.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np


def _sigmoid_platt(f: np.ndarray, A: float, B: float) -> np.ndarray:
    """Numerically stable 1 / (1 + exp(A * f + B))."""
    fApB = A * np.asarray(f, dtype=np.float64) + B
    out = np.empty_like(fApB)
    pos = fApB >= 0.0
    exp_neg = np.exp(-fApB[pos])
    out[pos] = exp_neg / (1.0 + exp_neg)
    out[~pos] = 1.0 / (1.0 + np.exp(fApB[~pos]))
    return out


def _platt_nll(f: np.ndarray, t: np.ndarray, A: float, B: float) -> float:
    """Numerically stable NLL sum_i [t_i * fApB_i + log(1 + exp(-fApB_i))] (Lin et al., 2007)."""
    fApB = A * f + B
    pos = fApB >= 0.0
    loss = np.empty_like(fApB)
    loss[pos] = t[pos] * fApB[pos] + np.log1p(np.exp(-fApB[pos]))
    loss[~pos] = (t[~pos] - 1.0) * fApB[~pos] + np.log1p(np.exp(fApB[~pos]))
    return float(np.sum(loss))


def fit_platt_sigmoid(
    decision_values: np.ndarray,
    y_binary: np.ndarray,
    max_iter: int = 100,
    min_step: float = 1e-10,
    sigma: float = 1e-12,
    eps: float = 1e-5,
) -> Tuple[float, float]:
    """Fit Platt sigmoid parameters (A, B) via Newton's method with backtracking (Lin et al., 2007).

    Parameters
    ----------
    decision_values : ndarray of shape (n_samples,)
        Out-of-fold SVM decision function values f(x_i).
    y_binary : ndarray of shape (n_samples,)
        Binary labels in {-1, +1} (or {0, 1}; positive class is > 0).

    Returns
    -------
    A, B : float
        Sigmoid parameters such that P(y=+1 | f) = 1 / (1 + exp(A * f + B)).
    """
    f = np.asarray(decision_values, dtype=np.float64)
    y_pos = np.asarray(y_binary) > 0
    n_pos = int(np.sum(y_pos))
    n_neg = int(f.size - n_pos)

    if n_pos == 0 or n_neg == 0:
        return 0.0, float(np.log((n_neg + 1.0) / (n_pos + 1.0)))

    # Regularized target probabilities (Platt, 1999)
    hi_target = (n_pos + 1.0) / (n_pos + 2.0)
    lo_target = 1.0 / (n_neg + 2.0)
    t = np.where(y_pos, hi_target, lo_target)

    # Initial point
    A = 0.0
    B = float(np.log((n_neg + 1.0) / (n_pos + 1.0)))
    fval = _platt_nll(f, t, A, B)

    for _ in range(max_iter):
        # Compute gradient (g1, g2) and damped Hessian (h11, h22, h21)
        p = _sigmoid_platt(f, A, B)
        d1 = t - p
        d2 = p * (1.0 - p)

        h11 = sigma + float(np.dot(f * f, d2))
        h22 = sigma + float(np.sum(d2))
        h21 = float(np.dot(f, d2))
        g1 = float(np.dot(f, d1))
        g2 = float(np.sum(d1))

        if abs(g1) < eps and abs(g2) < eps:
            break

        # Newton direction: -H^{-1} g
        det = h11 * h22 - h21 * h21
        if abs(det) < 1e-30:
            break
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        stepsize = 1.0
        while stepsize >= min_step:
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            newf = _platt_nll(f, t, newA, newB)
            if newf < fval + 1e-4 * stepsize * gd:
                A, B, fval = newA, newB, newf
                break
            stepsize *= 0.5
        else:
            break

    return float(A), float(B)


def _stratified_kfold_indices(
    y: np.ndarray, n_splits: int, random_state: Optional[int] = None
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Pure NumPy stratified K-fold splitter (no sklearn dependency required)."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    classes = np.unique(y)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for cls in classes:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for pos, sample_idx in enumerate(idx):
            folds[pos % n_splits].append(int(sample_idx))

    all_idx = np.arange(y.size)
    splits = []
    for k in range(n_splits):
        val_idx = np.array(folds[k], dtype=int)
        mask = np.ones(y.size, dtype=bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        splits.append((train_idx, val_idx))
    return splits


class PlattCalibrator:
    """Fits Platt scaling using internal stratified K-fold cross-validation."""

    def __init__(self, cv: int = 5, random_state: Optional[int] = 42) -> None:
        self.cv = cv
        self.random_state = random_state
        self.A_: float = 0.0
        self.B_: float = 0.0

    def fit(
        self,
        estimator_factory: Callable[[], object],
        X: np.ndarray,
        y: np.ndarray,
    ) -> "PlattCalibrator":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("PlattCalibrator expects binary labels.")
        y_bin = np.where(y == classes[1], 1.0, -1.0)

        min_class_count = int(min(np.sum(y_bin > 0), np.sum(y_bin < 0)))
        n_splits = min(int(self.cv), min_class_count)

        if n_splits >= 2:
            oof_scores = np.zeros(X.shape[0], dtype=np.float64)
            splits = _stratified_kfold_indices(y_bin, n_splits, self.random_state)
            for tr_idx, va_idx in splits:
                est = estimator_factory()
                est.fit(X[tr_idx], y[tr_idx])
                oof_scores[va_idx] = est.decision_function(X[va_idx])
        else:
            est = estimator_factory()
            est.fit(X, y)
            oof_scores = est.decision_function(X)

        self.A_, self.B_ = fit_platt_sigmoid(oof_scores, y_bin)
        return self

    def predict_proba(self, decision_values: np.ndarray) -> np.ndarray:
        """Return calibrated P(y = classes_[1] | f)."""
        return _sigmoid_platt(decision_values, self.A_, self.B_)


def pairwise_coupling(
    R: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> np.ndarray:
    """Estimate K-class probabilities from pairwise probability matrix R (Wu, Lin & Weng, 2004).

    Parameters
    ----------
    R : ndarray of shape (K, K)
        R[i, j] = P(y = i | y in {i, j}, x) for i != j, with R[j, i] = 1 - R[i, j].
    max_iter : int, default=1000
        Maximum iterations of Algorithm 2.
    tol : float, default=1e-7
        Stopping tolerance on optimality condition max_t |(Qp)_t - p^T Q p|.

    Returns
    -------
    p : ndarray of shape (K,)
        Estimated class probability vector summing to 1.
    """
    R = np.asarray(R, dtype=np.float64)
    K = R.shape[0]
    Q = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                Q[i, i] = float(np.sum(R[:, i] ** 2) - R[i, i] ** 2)
            else:
                Q[i, j] = -R[j, i] * R[i, j]

    p = np.full(K, 1.0 / K, dtype=np.float64)
    eps = 0.005 / K

    for _ in range(max_iter):
        Qp = Q @ p
        pQp = float(np.dot(p, Qp))
        max_diff = float(np.max(np.abs(Qp - pQp)))
        if max_diff < max(tol, eps * 1e-4):
            break
        for t in range(K):
            if Q[t, t] <= 0.0:
                continue
            diff = (-Qp[t] + pQp) / Q[t, t]
            p[t] += diff
            pQp = (pQp + diff * (diff * Q[t, t] + 2.0 * Qp[t])) / ((1.0 + diff) ** 2)
            p /= 1.0 + diff
            Qp = (Qp + diff * Q[:, t]) / (1.0 + diff)

    p = np.clip(p, 0.0, 1.0)
    total = float(np.sum(p))
    return p / total if total > 0.0 else np.full(K, 1.0 / K)


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    """Compute multiclass Brier score: (1/n) * sum_i sum_k (p_{ik} - 1[y_i == k])^2."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    Y_onehot = (y_true[:, None] == classes[None, :]).astype(np.float64)
    return float(np.mean(np.sum((y_proba - Y_onehot) ** 2, axis=1)))


def reliability_diagram_data(
    y_true_binary: np.ndarray,
    y_prob_pos: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean predicted probability, empirical positive rate, and bin counts."""
    y_true = (np.asarray(y_true_binary) > 0).astype(np.float64)
    y_prob = np.asarray(y_prob_pos, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)

    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_pred.append(float(np.mean(y_prob[mask])))
            frac_pos.append(float(np.mean(y_true[mask])))
            counts.append(cnt)
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)
