"""Positive-definite kernel functions and Gram matrix utilities.

Conventions match scikit-learn / LIBSVM so that A4 verification matches:
- linear: k(x, z) = x^T z
- poly:   k(x, z) = (gamma * x^T z + coef0) ** degree
- rbf:    k(x, z) = exp(-gamma * ||x - z||^2)
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np

GammaType = Union[float, str]


def resolve_gamma(gamma: GammaType, X: np.ndarray) -> float:
    """Resolve gamma parameter ('scale', 'auto', or positive float)."""
    X = np.asarray(X, dtype=np.float64)
    if isinstance(gamma, str):
        n_features = X.shape[1]
        if gamma == "scale":
            var = float(X.var())
            return 1.0 / (n_features * var) if var > 0.0 else 1.0
        if gamma == "auto":
            return 1.0 / n_features
        raise ValueError(f"Unsupported gamma string: {gamma!r}. Use 'scale', 'auto', or float.")
    val = float(gamma)
    if val <= 0.0:
        raise ValueError(f"gamma must be positive, got {val}.")
    return val


def linear_kernel(X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
    """Compute linear kernel matrix K(X, Z) = X @ Z.T."""
    X = np.asarray(X, dtype=np.float64)
    Z = X if Z is None else np.asarray(Z, dtype=np.float64)
    return X @ Z.T


def polynomial_kernel(
    X: np.ndarray,
    Z: np.ndarray | None = None,
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Compute polynomial kernel matrix K(X, Z) = (gamma * X @ Z.T + coef0) ** degree."""
    X = np.asarray(X, dtype=np.float64)
    Z = X if Z is None else np.asarray(Z, dtype=np.float64)
    return (gamma * (X @ Z.T) + coef0) ** degree


def rbf_kernel(
    X: np.ndarray,
    Z: np.ndarray | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute Gaussian (RBF) kernel matrix exp(-gamma * ||x - z||^2) via ||x||^2 + ||z||^2 - 2 x^T z."""
    X = np.asarray(X, dtype=np.float64)
    Z = X if Z is None else np.asarray(Z, dtype=np.float64)
    X_sq = np.einsum("ij,ij->i", X, X)[:, None]
    Z_sq = X_sq.T if Z is X else np.einsum("ij,ij->i", Z, Z)[None, :]
    sq_dists = X_sq + Z_sq - 2.0 * (X @ Z.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-gamma * sq_dists)


def compute_kernel_matrix(
    X: np.ndarray,
    Z: np.ndarray | None = None,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Dispatch vectorized Gram / cross-kernel matrix computation."""
    if kernel == "linear":
        return linear_kernel(X, Z)
    if kernel == "poly":
        return polynomial_kernel(X, Z, gamma=gamma, degree=degree, coef0=coef0)
    if kernel == "rbf":
        return rbf_kernel(X, Z, gamma=gamma)
    raise ValueError(f"Unsupported kernel: {kernel!r}. Choose from 'linear', 'poly', 'rbf'.")


def compute_kernel_diagonal(
    X: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Compute diagonal entries K[i, i] in O(n * d) without allocating the full Gram matrix."""
    X = np.asarray(X, dtype=np.float64)
    if kernel == "rbf":
        return np.ones(X.shape[0], dtype=np.float64)
    row_sq = np.einsum("ij,ij->i", X, X)
    if kernel == "linear":
        return row_sq
    if kernel == "poly":
        return (gamma * row_sq + coef0) ** degree
    raise ValueError(f"Unsupported kernel: {kernel!r}.")


def compute_kernel_column(
    X: np.ndarray,
    idx: int,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    X_sq: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a single Gram matrix column K[:, idx] in O(n * d) on the fly."""
    x_i = X[idx]
    dots = X @ x_i
    if kernel == "linear":
        return dots
    if kernel == "poly":
        return (gamma * dots + coef0) ** degree
    if kernel == "rbf":
        if X_sq is None:
            X_sq = np.einsum("ij,ij->i", X, X)
        sq_dists = X_sq + X_sq[idx] - 2.0 * dots
        np.maximum(sq_dists, 0.0, out=sq_dists)
        return np.exp(-gamma * sq_dists)
    raise ValueError(f"Unsupported kernel: {kernel!r}.")


def is_psd(K: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    """Check whether a symmetric matrix K is positive semi-definite within tolerance."""
    K = np.asarray(K, dtype=np.float64)
    K_sym = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(K_sym)
    min_eig = float(eigvals[0])
    return (min_eig >= -tol), min_eig
