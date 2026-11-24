"""Sequential Minimal Optimization (SMO) solver for Soft-Margin SVM.

Implements:
- Maximal KKT violating pair selection (WSS1; Keerthi et al., 2001)
- Analytic 2-variable update (Platt, 1998) with mandatory boundary rounding (1e-12 * C)
- Incremental O(n) gradient updates using only 2 kernel columns per iteration
- Support for precomputed Gram matrix (fast for n <= 10^4) and on-the-fly kernel
  columns (memory-safe for large n up to 10^5 in C5)
- Optional automatic multiclass (OvO / OvR) and Platt probability calibration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np

from ._base import BaseEstimator, ClassifierMixin
from .kernels import (
    compute_kernel_column,
    compute_kernel_diagonal,
    compute_kernel_matrix,
    resolve_gamma,
)


class BinarySMO(BaseEstimator, ClassifierMixin):
    """Binary Soft-Margin SVM trained with SMO (WSS1).

    Parameters
    ----------
    C : float, default=1.0
        Regularization (penalty) parameter.
    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        Kernel type.
    gamma : float or {'scale', 'auto'}, default=1.0
        Kernel coefficient for 'rbf' and 'poly'.
    degree : int, default=3
        Degree for 'poly' kernel.
    coef0 : float, default=1.0
        Independent term in 'poly' kernel.
    tol : float, default=1e-3
        KKT stopping tolerance: stop when m(alpha) - M(alpha) < tol.
    max_iter : int, default=100_000
        Maximum number of SMO pairwise updates.
    cache_kernel : bool or 'auto', default='auto'
        If True, precompute full n x n Gram matrix. If False, compute kernel
        columns on the fly (O(n) memory). If 'auto', precompute when n <= 8000.
    record_history : bool, default=False
        If True, log dual objective D(alpha) and KKT gap m(alpha) - M(alpha).
    history_stride : int, default=1
        Record history every `history_stride` iterations.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[float, str] = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 100_000,
        cache_kernel: Union[bool, str] = "auto",
        record_history: bool = False,
        history_stride: int = 1,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.cache_kernel = cache_kernel
        self.record_history = record_history
        self.history_stride = history_stride

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinarySMO":
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if y_raw.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        classes = np.unique(y_raw)
        if classes.size != 2:
            raise ValueError(f"BinarySMO requires exactly 2 classes, got {classes.size}.")
        self.classes_ = classes
        # Map classes_[0] -> -1, classes_[1] -> +1 (matches sklearn.svm.SVC)
        y_bin = np.where(y_raw == classes[1], 1.0, -1.0)

        n = X.shape[0]
        C = float(self.C)
        if C <= 0.0:
            raise ValueError(f"C must be positive, got {C}.")
        gamma_val = resolve_gamma(self.gamma, X)
        self.gamma_ = gamma_val

        use_cache = (
            (n <= 8000) if self.cache_kernel == "auto" else bool(self.cache_kernel)
        )

        if use_cache:
            K = compute_kernel_matrix(
                X,
                kernel=self.kernel,
                gamma=gamma_val,
                degree=self.degree,
                coef0=self.coef0,
            )
            K_diag = np.diag(K)
            X_sq = None
        else:
            K = None
            X_sq = np.einsum("ij,ij->i", X, X) if self.kernel == "rbf" else None
            K_diag = compute_kernel_diagonal(
                X,
                kernel=self.kernel,
                gamma=gamma_val,
                degree=self.degree,
                coef0=self.coef0,
            )

        alpha = np.zeros(n, dtype=np.float64)
        G = -np.ones(n, dtype=np.float64)  # G = Q @ alpha - e
        eps_bound = 1e-12 * C
        y_pos = y_bin > 0.0
        y_neg = ~y_pos

        history: Dict[str, List[float]] = {"iter": [], "dual_obj": [], "kkt_gap": []}
        converged = False
        n_iter = 0
        m_val = np.inf
        M_val = -np.inf

        for it in range(1, self.max_iter + 1):
            # I_up = {t : (y_t=+1 & alpha_t < C) or (y_t=-1 & alpha_t > 0)}
            # I_low = {t : (y_t=+1 & alpha_t > 0) or (y_t=-1 & alpha_t < C)}
            i_up_mask = (y_pos & (alpha < C)) | (y_neg & (alpha > 0.0))
            i_low_mask = (y_pos & (alpha > 0.0)) | (y_neg & (alpha < C))

            neg_yG = -y_bin * G

            # Maximal violating pair (WSS1)
            up_vals = np.where(i_up_mask, neg_yG, -np.inf)
            low_vals = np.where(i_low_mask, neg_yG, np.inf)
            i = int(np.argmax(up_vals))
            j = int(np.argmin(low_vals))
            m_val = float(up_vals[i])
            M_val = float(low_vals[j])
            gap = m_val - M_val

            if self.record_history and ((it - 1) % self.history_stride == 0 or gap < self.tol):
                dual_obj_now = -0.5 * float(np.dot(alpha, G - 1.0))
                history["iter"].append(it - 1)
                history["dual_obj"].append(dual_obj_now)
                history["kkt_gap"].append(gap)

            if gap < self.tol:
                converged = True
                break

            n_iter = it
            ai_old = alpha[i]
            aj_old = alpha[j]
            yi = y_bin[i]
            yj = y_bin[j]

            # Analytic box constraints [L, H] for alpha_j
            if yi != yj:
                L = max(0.0, aj_old - ai_old)
                H = min(C, C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - C)
                H = min(C, ai_old + aj_old)

            if L >= H:
                break

            # Kernel columns K[:, i] and K[:, j]
            if K is not None:
                K_i = K[:, i]
                K_j = K[:, j]
                K_ij = K_i[j]
            else:
                K_i = compute_kernel_column(
                    X, i, kernel=self.kernel, gamma=gamma_val,
                    degree=self.degree, coef0=self.coef0, X_sq=X_sq,
                )
                K_j = compute_kernel_column(
                    X, j, kernel=self.kernel, gamma=gamma_val,
                    degree=self.degree, coef0=self.coef0, X_sq=X_sq,
                )
                K_ij = K_i[j]

            eta = K_diag[i] + K_diag[j] - 2.0 * K_ij
            if eta <= 0.0:
                eta = 1e-12

            # Analytic 2-variable update
            aj_new = aj_old + yj * (yi * G[i] - yj * G[j]) / eta
            if aj_new > H:
                aj_new = H
            elif aj_new < L:
                aj_new = L
            ai_new = ai_old + yi * yj * (aj_old - aj_new)

            # Mandatory boundary rounding (prevents 1e-17 stagnation)
            if ai_new < eps_bound:
                ai_new = 0.0
            elif ai_new > C - eps_bound:
                ai_new = C
            if aj_new < eps_bound:
                aj_new = 0.0
            elif aj_new > C - eps_bound:
                aj_new = C

            d_ai = ai_new - ai_old
            d_aj = aj_new - aj_old
            if d_ai == 0.0 and d_aj == 0.0:
                break

            alpha[i] = ai_new
            alpha[j] = aj_new

            # Incremental O(n) gradient update
            G += y_bin * (yi * d_ai * K_i + yj * d_aj * K_j)

        # Compute bias b
        free_mask = (alpha > 0.0) & (alpha < C)
        if np.any(free_mask):
            b = float(np.mean(-y_bin[free_mask] * G[free_mask]))
        else:
            b = 0.5 * (m_val + M_val)

        # Dual objective D(alpha) = sum(alpha) - 0.5 * alpha^T Q alpha = -0.5 * sum(alpha * (G - 1))
        dual_obj = -0.5 * float(np.dot(alpha, G - 1.0))

        # Support vectors
        sv_tol = 1e-12 * C
        support = np.flatnonzero(alpha > sv_tol)

        self.alpha_ = alpha
        self.support_ = support
        self.support_vectors_ = X[support].copy()
        self.dual_coef_ = (alpha[support] * y_bin[support])[None, :]
        self.b_ = b
        self.intercept_ = np.array([b], dtype=np.float64)
        self.dual_objective_ = dual_obj
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.kkt_gap_ = m_val - M_val
        self.history_ = history

        if self.kernel == "linear":
            self.coef_ = (alpha[support] * y_bin[support]) @ X[support]
        else:
            self.coef_ = None

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute signed distance / decision values f(x) = sum_s alpha_s y_s k(x_s, x) + b."""
        X = np.asarray(X, dtype=np.float64)
        if self.kernel == "linear" and self.coef_ is not None:
            return X @ self.coef_ + self.b_
        if self.support_.size == 0:
            return np.full(X.shape[0], self.b_, dtype=np.float64)
        K_test = compute_kernel_matrix(
            X,
            self.support_vectors_,
            kernel=self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )
        return K_test @ self.dual_coef_[0] + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])


class SVC(BaseEstimator, ClassifierMixin):
    """Unified Support Vector Classifier (Binary SMO + Multiclass OvO/OvR + Platt Scaling).

    Matches the team's agreed interface while transparently supporting multiclass
    datasets (e.g. Iris, Wine, and P3's 5-variety Rice dataset) and `predict_proba`.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[float, str] = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 100_000,
        cache_kernel: Union[bool, str] = "auto",
        decision_function_shape: str = "ovo",
        probability: bool = False,
        cv: int = 5,
        random_state: Optional[int] = None,
        record_history: bool = False,
        history_stride: int = 1,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.cache_kernel = cache_kernel
        self.decision_function_shape = decision_function_shape
        self.probability = probability
        self.cv = cv
        self.random_state = random_state
        self.record_history = record_history
        self.history_stride = history_stride

    def _make_binary_estimator(self) -> BinarySMO:
        return BinarySMO(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=self.max_iter,
            cache_kernel=self.cache_kernel,
            record_history=self.record_history,
            history_stride=self.history_stride,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.size
        if n_classes < 2:
            raise ValueError(f"SVC requires at least 2 classes, got {n_classes}.")

        if n_classes == 2:
            self._is_multiclass = False
            est = self._make_binary_estimator().fit(X, y)
            self._estimator = est
            # Expose required binary attributes directly on SVC
            self.alpha_ = est.alpha_
            self.support_ = est.support_
            self.support_vectors_ = est.support_vectors_
            self.dual_coef_ = est.dual_coef_
            self.b_ = est.b_
            self.intercept_ = est.intercept_
            self.dual_objective_ = est.dual_objective_
            self.n_iter_ = est.n_iter_
            self.converged_ = est.converged_
            self.kkt_gap_ = est.kkt_gap_
            self.history_ = est.history_
            self.gamma_ = est.gamma_
            self.coef_ = est.coef_

            if self.probability:
                from .platt import PlattCalibrator

                self._platt = PlattCalibrator(cv=self.cv, random_state=self.random_state)
                self._platt.fit(self._make_binary_estimator, X, y)
            else:
                self._platt = None
        else:
            from .multiclass import OneVsOneClassifier, OneVsRestClassifier

            self._is_multiclass = True
            shape = self.decision_function_shape.lower()
            if shape == "ovo":
                mc = OneVsOneClassifier(
                    estimator_factory=self._make_binary_estimator,
                    probability=self.probability,
                    cv=self.cv,
                    random_state=self.random_state,
                )
            elif shape == "ovr":
                mc = OneVsRestClassifier(
                    estimator_factory=self._make_binary_estimator,
                    probability=self.probability,
                    cv=self.cv,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(
                    f"decision_function_shape must be 'ovo' or 'ovr', got {self.decision_function_shape!r}."
                )
            mc.fit(X, y)
            self._estimator = mc
            self.support_ = mc.support_
            self.gamma_ = resolve_gamma(self.gamma, X)
            self.n_iter_ = sum(e.n_iter_ for e in mc.estimators_)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise RuntimeError("predict_proba is unavailable when probability=False.")
        if not self._is_multiclass:
            scores = self._estimator.decision_function(X)
            p_pos = self._platt.predict_proba(scores)
            return np.column_stack([1.0 - p_pos, p_pos])
        return self._estimator.predict_proba(X)
