"""A4 Verification Suite: svmlib vs LIBSVM (sklearn.svm.SVC), LinearSVC, and KernelRidge.

Run on Google Colab or local Python environment with:
    pytest -v tests/test_vs_libsvm.py
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification, make_moons
from sklearn.kernel_ridge import KernelRidge as SkKernelRidge
from sklearn.metrics.pairwise import linear_kernel as sk_linear
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC as SkLinearSVC
from sklearn.svm import SVC as SkSVC

from svmlib import KernelRidge, LinearSVMDCD, SVC, compute_kernel_matrix, is_psd
from svmlib.platt import brier_score_multiclass


def _sklearn_dual_objective(sk_model: SkSVC, X: np.ndarray, kernel: str, gamma: float, degree: int, coef0: float) -> float:
    """Compute dual objective D(alpha) = sum(|dual_coef_|) - 0.5 * dual_coef_ K_sv dual_coef_^T."""
    sv_idx = sk_model.support_
    dc = sk_model.dual_coef_[0]  # y_s * alpha_s
    K_sv = compute_kernel_matrix(
        X[sv_idx], kernel=kernel, gamma=gamma, degree=degree, coef0=coef0
    )
    return float(np.sum(np.abs(dc)) - 0.5 * (dc @ K_sv @ dc))


# ---------------------------------------------------------------------------
# 1. Kernels vs sklearn.metrics.pairwise & PSD property
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
def test_kernels_match_sklearn_and_psd(kernel: str):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 6))
    Z = rng.normal(size=(30, 6))
    gamma, degree, coef0 = 0.35, 3, 1.0

    K_ours = compute_kernel_matrix(X, Z, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    if kernel == "linear":
        K_sk = sk_linear(X, Z)
    elif kernel == "poly":
        K_sk = sk_poly(X, Z, degree=degree, gamma=gamma, coef0=coef0)
    else:
        K_sk = sk_rbf(X, Z, gamma=gamma)

    assert np.allclose(K_ours, K_sk, atol=1e-12)

    K_gram = compute_kernel_matrix(X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    ok, min_eig = is_psd(K_gram, tol=1e-10)
    assert ok, f"{kernel} Gram matrix failed PSD check: min_eig={min_eig}"


# ---------------------------------------------------------------------------
# 2. C1 analytical verification
# ---------------------------------------------------------------------------
def test_c1_analytical_solution():
    X = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    y = np.array([1, 1, -1])
    model = SVC(C=1e6, kernel="linear", tol=1e-9).fit(X, y)

    assert np.allclose(model.alpha_, [0.5, 0.5, 1.0], atol=1e-6)
    assert np.allclose(model.coef_, [1.0, 1.0], atol=1e-6)
    assert abs(model.b_ - (-1.0)) < 1e-6
    assert abs(model.dual_objective_ - 1.0) < 1e-6
    assert set(model.support_.tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# 3. Binary SMO vs LIBSVM (sklearn.svm.SVC) across datasets, kernels, and C
# ---------------------------------------------------------------------------
def _binary_datasets():
    X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

    bc = load_breast_cancer()
    X_bc = StandardScaler().fit_transform(bc.data)
    y_bc = bc.target

    iris = load_iris()
    mask = iris.target < 2
    X_iris = StandardScaler().fit_transform(iris.data[mask])
    y_iris = iris.target[mask]

    X_syn, y_syn = make_classification(
        n_samples=180, n_features=8, n_informative=5, random_state=7
    )
    X_syn = StandardScaler().fit_transform(X_syn)

    return [
        ("moons", X_moons, y_moons, 1.0),
        ("breast_cancer", X_bc, y_bc, 0.05),
        ("iris_binary", X_iris, y_iris, 0.5),
        ("synthetic", X_syn, y_syn, 0.2),
    ]


@pytest.mark.parametrize("ds_name,X,y,gamma", _binary_datasets())
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_binary_smo_vs_libsvm(ds_name, X, y, gamma, kernel, C):
    tol = 1e-6
    degree, coef0 = 3, 1.0

    ours = SVC(
        C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
        tol=tol, max_iter=200_000,
    ).fit(X, y)
    sk = SkSVC(
        C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, tol=tol,
    ).fit(X, y)

    # Dual objective relative error < 1e-4
    D_sk = _sklearn_dual_objective(sk, X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    rel_diff = abs(ours.dual_objective_ - D_sk) / max(1.0, abs(D_sk))
    assert rel_diff < 1e-4, f"{ds_name}/{kernel}/C={C}: rel_diff={rel_diff}"

    # Predictions match 100%
    assert np.array_equal(ours.predict(X), sk.predict(X))

    # Decision function & bias b match closely
    max_df_diff = float(np.max(np.abs(ours.decision_function(X) - sk.decision_function(X))))
    assert max_df_diff < 5e-3, f"max |Delta f| = {max_df_diff}"
    assert abs(ours.b_ - float(sk.intercept_[0])) < 5e-3

    # Support vector sets match for non-borderline SVs (alpha > 1e-6 * C)
    sv_ours = set(np.flatnonzero(ours.alpha_ > 1e-6 * C).tolist())
    sv_sk = set(sk.support_[np.abs(sk.dual_coef_[0]) > 1e-6 * C].tolist())
    # Allow at most tiny boundary discrepancy on degenerate points
    sym_diff = len(sv_ours ^ sv_sk)
    assert sym_diff <= max(2, int(0.02 * len(sv_sk))), f"SV diff={sym_diff}"


# ---------------------------------------------------------------------------
# 4. DCD vs sklearn LinearSVC(loss="hinge", dual=True)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_dcd_vs_linear_svc(C: float):
    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    y = bc.target

    dcd = LinearSVMDCD(C=C, tol=1e-5, max_iter=50_000, fit_intercept=True, random_state=42).fit(X, y)
    sk = SkLinearSVC(C=C, loss="hinge", dual=True, tol=1e-5, max_iter=50_000, random_state=42).fit(X, y)

    # Compare primal objective (including regularized bias)
    y_bin = np.where(y == 1, 1.0, -1.0)
    p_dcd = 0.5 * (np.dot(dcd.coef_, dcd.coef_) + dcd.b_ ** 2) + C * np.sum(np.maximum(0.0, 1.0 - y_bin * dcd.decision_function(X)))
    p_sk = 0.5 * (np.dot(sk.coef_[0], sk.coef_[0]) + sk.intercept_[0] ** 2) + C * np.sum(np.maximum(0.0, 1.0 - y_bin * sk.decision_function(X)))
    rel_obj_diff = abs(p_dcd - p_sk) / max(1.0, abs(p_sk))
    assert rel_obj_diff < 1e-3
    assert np.mean(dcd.predict(X) == sk.predict(X)) >= 0.99


# ---------------------------------------------------------------------------
# 5. Kernel Ridge Regression vs sklearn.kernel_ridge.KernelRidge
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
def test_krr_vs_sklearn(kernel: str):
    rng = np.random.default_rng(123)
    X_tr = rng.normal(size=(80, 4))
    y_tr = np.sin(X_tr[:, 0]) + 0.5 * X_tr[:, 1] ** 2 - 0.3 * X_tr[:, 2]
    X_te = rng.normal(size=(30, 4))

    ours = KernelRidge(alpha=0.5, kernel=kernel, gamma=0.5, degree=3, coef0=1.0).fit(X_tr, y_tr)
    sk = SkKernelRidge(alpha=0.5, kernel=kernel, gamma=0.5, degree=3, coef0=1.0).fit(X_tr, y_tr)

    assert np.max(np.abs(ours.dual_coef_ - sk.dual_coef_)) < 1e-8
    assert np.max(np.abs(ours.predict(X_te) - sk.predict(X_te))) < 1e-8


# ---------------------------------------------------------------------------
# 6. Multiclass (OvO & OvR) vs LIBSVM + Platt Scaling & Pipeline Compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dataset_loader", [load_iris, load_wine])
def test_multiclass_ovo_matches_libsvm_and_pipeline(dataset_loader):
    data = dataset_loader()
    X, y = data.data, data.target

    pipe_ours = make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", gamma=0.2, tol=1e-6, decision_function_shape="ovo", probability=True, random_state=42),
    )
    pipe_sk = make_pipeline(
        StandardScaler(),
        SkSVC(C=1.0, kernel="rbf", gamma=0.2, tol=1e-6, decision_function_shape="ovo"),
    )

    pipe_ours.fit(X, y)
    pipe_sk.fit(X, y)

    # OvO predictions match LIBSVM
    assert np.array_equal(pipe_ours.predict(X), pipe_sk.predict(X))

    # Calibrated probabilities valid & well-calibrated (low Brier score)
    proba = pipe_ours.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    brier = brier_score_multiclass(y, proba, np.unique(y))
    assert brier < 0.15

    # OvR also works cleanly with valid probabilities
    pipe_ovr = make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", gamma=0.2, tol=1e-6, decision_function_shape="ovr", probability=True, random_state=42),
    ).fit(X, y)
    proba_ovr = pipe_ovr.predict_proba(X)
    assert np.allclose(proba_ovr.sum(axis=1), 1.0, atol=1e-9)
    assert pipe_ovr.score(X, y) > 0.90
