"""Kernel Ridge Regression (KRR) with closed-form solution.

Solves:
    (K + alpha * I) dual_coef = y
using Cholesky factorization (scipy.linalg.cho_factor / cho_solve) with
fallback to np.linalg.solve — never computing an explicit matrix inverse.
"""

from __future__ import annotations

from typing import Union
import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from ._base import BaseEstimator, RegressorMixin
from .kernels import compute_kernel_matrix, resolve_gamma


class KernelRidge(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regression matching sklearn.kernel_ridge.KernelRidge.

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularization strength (lambda in the problem statement).
    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        Kernel function.
    gamma : float or {'scale', 'auto'}, default=1.0
        Kernel coefficient for 'rbf' and 'poly'.
    degree : int, default=3
        Degree for 'poly' kernel.
    coef0 : float, default=1.0
        Independent term for 'poly' kernel.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[float, str] = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidge":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        lam = float(self.alpha)
        if lam < 0.0:
            raise ValueError(f"alpha (lambda) must be non-negative, got {lam}.")

        self.X_fit_ = X.copy()
        self.gamma_ = resolve_gamma(self.gamma, X)
        K = compute_kernel_matrix(
            X,
            kernel=self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )
        A = K.copy()
        A[np.diag_indices_from(A)] += lam

        try:
            c, lower = cho_factor(A, overwrite_a=False, check_finite=False)
            dual_coef = cho_solve((c, lower), y, check_finite=False)
        except LinAlgError:
            dual_coef = np.linalg.solve(A, y)

        self.dual_coef_ = dual_coef
        self.alpha_ = dual_coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        K_test = compute_kernel_matrix(
            X,
            self.X_fit_,
            kernel=self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )
        return K_test @ self.dual_coef_
