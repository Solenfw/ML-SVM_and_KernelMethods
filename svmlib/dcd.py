"""Dual Coordinate Descent (DCD) with Shrinking for Large-Scale Linear SVM (Hsieh et al., 2008).

Solves the L1-loss (hinge) dual linear SVM (Algorithm 3 of Hsieh et al., 2008 / LIBLINEAR):
    min_alpha  0.5 * ||sum_i alpha_i y_i x_tilde_i||^2 - sum_i alpha_i
    s.t.       0 <= alpha_i <= C

Bias handling:
    When `fit_intercept=True`, augments each sample with a constant feature
    `intercept_scaling` (default 1.0), matching LIBLINEAR / sklearn's
    `LinearSVC(loss='hinge', dual=True)`. Note that this regularizes `b`
    (adds 0.5 * b^2 / intercept_scaling^2 to the primal), unlike SMO.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from ._base import BaseEstimator, ClassifierMixin

try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


if _HAS_NUMBA:  # pragma: no cover

    @_njit(cache=True)
    def _dcd_loop_numba(
        X_aug: np.ndarray,
        y: np.ndarray,
        Q_diag: np.ndarray,
        C: float,
        tol: float,
        max_iter: int,
        seed: int,
    ):
        n, d = X_aug.shape
        alpha = np.zeros(n, dtype=np.float64)
        w = np.zeros(d, dtype=np.float64)
        index = np.arange(n)
        active_size = n
        np.random.seed(seed)

        PG_max_old = np.inf
        PG_min_old = -np.inf
        converged = False
        n_iter = 0
        pg_gap = np.inf

        for epoch in range(1, max_iter + 1):
            PG_max_new = -np.inf
            PG_min_new = np.inf

            # Random permutation of active set
            for k in range(active_size):
                r = k + np.random.randint(0, active_size - k)
                tmp = index[k]
                index[k] = index[r]
                index[r] = tmp

            s = 0
            while s < active_size:
                i = index[s]
                yi = y[i]
                dot = 0.0
                for j in range(d):
                    dot += w[j] * X_aug[i, j]
                G = yi * dot - 1.0

                ai = alpha[i]
                PG = 0.0
                if ai == 0.0:
                    if G > PG_max_old:
                        active_size -= 1
                        tmp = index[s]
                        index[s] = index[active_size]
                        index[active_size] = tmp
                        continue
                    elif G < 0.0:
                        PG = G
                elif ai == C:
                    if G < PG_min_old:
                        active_size -= 1
                        tmp = index[s]
                        index[s] = index[active_size]
                        index[active_size] = tmp
                        continue
                    elif G > 0.0:
                        PG = G
                else:
                    PG = G

                if PG > PG_max_new:
                    PG_max_new = PG
                if PG < PG_min_new:
                    PG_min_new = PG

                if abs(PG) > 1e-12 and Q_diag[i] > 0.0:
                    ai_new = ai - G / Q_diag[i]
                    if ai_new < 0.0:
                        ai_new = 0.0
                    elif ai_new > C:
                        ai_new = C
                    delta = ai_new - ai
                    if delta != 0.0:
                        alpha[i] = ai_new
                        step = delta * yi
                        for j in range(d):
                            w[j] += step * X_aug[i, j]
                s += 1

            n_iter = epoch
            pg_gap = PG_max_new - PG_min_new
            if pg_gap <= tol:
                if active_size == n:
                    converged = True
                    break
                else:
                    active_size = n
                    PG_max_old = np.inf
                    PG_min_old = -np.inf
                    continue

            PG_max_old = PG_max_new if PG_max_new > 0.0 else np.inf
            PG_min_old = PG_min_new if PG_min_new < 0.0 else -np.inf

        return alpha, w, n_iter, converged, pg_gap


def _dcd_loop_numpy(
    X_aug: np.ndarray,
    y: np.ndarray,
    Q_diag: np.ndarray,
    C: float,
    tol: float,
    max_iter: int,
    rng: np.random.Generator,
):
    """Pure NumPy/Python DCD loop with Hsieh et al. (2008) active-set shrinking."""
    n, d = X_aug.shape
    alpha = np.zeros(n, dtype=np.float64)
    w = np.zeros(d, dtype=np.float64)
    yX = X_aug * y[:, None]
    inv_Q = np.where(Q_diag > 0.0, 1.0 / Q_diag, 0.0)

    index = np.arange(n, dtype=np.int64)
    active_size = n
    PG_max_old = np.inf
    PG_min_old = -np.inf

    converged = False
    n_iter = 0
    pg_gap = np.inf

    for epoch in range(1, max_iter + 1):
        PG_max_new = -np.inf
        PG_min_new = np.inf

        rng.shuffle(index[:active_size])

        s = 0
        while s < active_size:
            i = int(index[s])
            ai = alpha[i]
            yxi = yX[i]
            G = float(np.dot(w, yxi)) - 1.0

            PG = 0.0
            if ai == 0.0:
                if G > PG_max_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
                elif G < 0.0:
                    PG = G
            elif ai == C:
                if G < PG_min_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
                elif G > 0.0:
                    PG = G
            else:
                PG = G

            if PG > PG_max_new:
                PG_max_new = PG
            if PG < PG_min_new:
                PG_min_new = PG

            if abs(PG) > 1e-12 and inv_Q[i] > 0.0:
                ai_new = ai - G * inv_Q[i]
                if ai_new < 0.0:
                    ai_new = 0.0
                elif ai_new > C:
                    ai_new = C
                delta = ai_new - ai
                if delta != 0.0:
                    alpha[i] = ai_new
                    w += delta * yxi
            s += 1

        n_iter = epoch
        pg_gap = PG_max_new - PG_min_new
        if pg_gap <= tol:
            if active_size == n:
                converged = True
                break
            else:
                active_size = n
                PG_max_old = np.inf
                PG_min_old = -np.inf
                continue

        PG_max_old = PG_max_new if PG_max_new > 0.0 else np.inf
        PG_min_old = PG_min_new if PG_min_new < 0.0 else -np.inf

    return alpha, w, n_iter, converged, pg_gap


class LinearSVMDCD(BaseEstimator, ClassifierMixin):
    """Linear SVM trained via Dual Coordinate Descent with Shrinking (LIBLINEAR).

    Parameters
    ----------
    C : float, default=1.0
        Penalty parameter for hinge loss.
    tol : float, default=1e-4
        Projected gradient stopping tolerance.
    max_iter : int, default=1000
        Maximum number of outer epochs (passes over the active set).
    fit_intercept : bool, default=True
        Whether to append a constant bias column to X.
    intercept_scaling : float, default=1.0
        Constant value appended when `fit_intercept=True` (matches LinearSVC).
    random_state : int or None, default=42
        Random seed for per-epoch coordinate permutation.
    use_numba : bool, default=True
        Use Numba JIT acceleration automatically when Numba is installed.
    """

    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        random_state: Optional[int] = 42,
        use_numba: bool = True,
    ) -> None:
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.use_numba = use_numba

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMDCD":
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y)
        classes = np.unique(y_raw)
        if classes.size != 2:
            raise ValueError(f"LinearSVMDCD requires 2 classes, got {classes.size}.")
        self.classes_ = classes
        y_bin = np.where(y_raw == classes[1], 1.0, -1.0)

        n, d = X.shape
        C = float(self.C)
        if C <= 0.0:
            raise ValueError(f"C must be positive, got {C}.")

        if self.fit_intercept:
            bias_col = np.full((n, 1), float(self.intercept_scaling), dtype=np.float64)
            X_aug = np.hstack([X, bias_col])
        else:
            X_aug = np.ascontiguousarray(X, dtype=np.float64)

        Q_diag = np.einsum("ij,ij->i", X_aug, X_aug)
        seed = 0 if self.random_state is None else int(self.random_state)

        if self.use_numba and _HAS_NUMBA:
            alpha, w_aug, n_iter, converged, pg_gap = _dcd_loop_numba(
                X_aug, y_bin, Q_diag, C, float(self.tol), int(self.max_iter), seed
            )
        else:
            rng = np.random.default_rng(self.random_state)
            alpha, w_aug, n_iter, converged, pg_gap = _dcd_loop_numpy(
                X_aug, y_bin, Q_diag, C, float(self.tol), int(self.max_iter), rng
            )

        self.alpha_ = alpha
        self.support_ = np.flatnonzero(alpha > 1e-12 * C)
        if self.fit_intercept:
            self.coef_ = w_aug[:d].copy()
            self.b_ = float(w_aug[d] * self.intercept_scaling)
        else:
            self.coef_ = w_aug.copy()
            self.b_ = 0.0
        self.intercept_ = np.array([self.b_], dtype=np.float64)
        self.dual_objective_ = float(np.sum(alpha) - 0.5 * np.dot(w_aug, w_aug))
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.pg_gap_ = pg_gap
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
