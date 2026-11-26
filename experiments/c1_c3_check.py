"""C1–C3: numeric verification of the hand-derived answers in report/sections/C_exercises.tex.

- C1: hard-margin SVM on 3 points -> alpha = (1/2, 1/2, 1), w = (1, 1), b = -1, D = 1, all KKT hold.
- C2: XOR is not linearly separable (LP feasibility); explicit phi reproduces (x^T z + 1)^2;
      SMO with the degree-2 kernel finds alpha_i = 1/8, b = 0, w = (0, 0, -1/sqrt2, 0, 0, 0).
- C3: k(x, z) = ||x - z|| has a Gram matrix with a negative eigenvalue.

Exits with a non-zero status if any check fails. Usage: python experiments/c1_c3_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from sklearn.svm import SVC as LibSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmlib import SVC

HARD_C = 1e6
TOL = 1e-6
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def phi(X: np.ndarray) -> np.ndarray:
    """Explicit feature map with phi(x)^T phi(z) = (x^T z + 1)^2 for x in R^2."""
    x1, x2 = X[:, 0], X[:, 1]
    s = np.sqrt(2.0)
    return np.column_stack([x1**2, x2**2, s * x1 * x2, s * x1, s * x2, np.ones(len(X))])


def c1() -> None:
    print("C1: hard-margin SVM on (2,0), (0,2) [+1] and (0,0) [-1]")
    X = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    y = np.array([1, 1, -1])
    m = SVC(C=HARD_C, kernel="linear", tol=1e-9).fit(X, y)
    alpha, w, b = m.alpha_, m.coef_.ravel(), m.b_
    f = X @ w + b
    check("alpha = (1/2, 1/2, 1)", np.allclose(alpha, [0.5, 0.5, 1.0], atol=TOL), f"alpha={alpha.round(8)}")
    check("w = (1, 1)", np.allclose(w, [1.0, 1.0], atol=TOL), f"w={w.round(8)}")
    check("b = -1", abs(b + 1.0) < TOL, f"b={b:.8f}")
    check("dual objective D = 1", abs(m.dual_objective_ - 1.0) < TOL, f"D={m.dual_objective_:.8f}")
    check("margin width 2/||w|| = sqrt(2)", abs(2 / np.linalg.norm(w) - np.sqrt(2)) < TOL)
    check("primal feasibility y f(x) >= 1", np.all(y * f >= 1 - TOL), f"y*f={np.round(y * f, 8)}")
    check("stationarity sum a_i y_i x_i = w", np.allclose((alpha * y) @ X, w, atol=TOL))
    check("sum a_i y_i = 0", abs(alpha @ y) < TOL)
    check("complementary slackness a_i (y_i f_i - 1) = 0", np.allclose(alpha * (y * f - 1), 0, atol=TOL))
    lib = LibSVC(C=HARD_C, kernel="linear", tol=1e-9).fit(X, y)
    check("LIBSVM gives the same w, b", np.allclose(lib.coef_.ravel(), w, atol=1e-5) and abs(lib.intercept_[0] - b) < 1e-5,
          f"libsvm w={lib.coef_.ravel().round(6)}, b={lib.intercept_[0]:.6f}")


def c2() -> None:
    print("C2: XOR and the degree-2 polynomial kernel")
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    y = np.array([-1, -1, 1, 1])  # y = -x1*x2  (XOR: coordinates differ -> +1)

    # Linear separability <=> LP {y_i (w^T x_i + b) >= 1} is feasible. Variables (w1, w2, b).
    A_ub = -y[:, None] * np.column_stack([X, np.ones(len(X))])
    lp = linprog(c=np.zeros(3), A_ub=A_ub, b_ub=-np.ones(len(X)), bounds=[(None, None)] * 3, method="highs")
    check("XOR is not linearly separable (LP infeasible)", lp.status == 2, f"linprog status={lp.status}: {lp.message}")

    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))
    check("phi(x)^T phi(z) = (x^T z + 1)^2 on random points",
          np.allclose(phi(A) @ phi(B).T, (A @ B.T + 1) ** 2, atol=1e-10))

    v = np.array([0, 0, -1 / np.sqrt(2), 0, 0, 0])
    check("hand-picked hyperplane separates XOR in feature space", np.all(y * (phi(X) @ v) > 0),
          f"y*(v^T phi)={y * (phi(X) @ v)}")

    m = SVC(C=HARD_C, kernel="poly", degree=2, gamma=1.0, coef0=1.0, tol=1e-9).fit(X, y)
    w_feat = (m.alpha_ * y) @ phi(X)
    check("SMO: alpha_i = 1/8 for all i", np.allclose(m.alpha_, 0.125, atol=TOL), f"alpha={m.alpha_.round(8)}")
    check("SMO: b = 0", abs(m.b_) < TOL, f"b={m.b_:.2e}")
    check("SMO: w in feature space = (0,0,-1/sqrt2,0,0,0)", np.allclose(w_feat, v, atol=TOL), f"w={w_feat.round(6)}")
    check("SMO: decision function = -x1*x2", np.allclose(m.decision_function(X), -X[:, 0] * X[:, 1], atol=TOL))
    check("SMO: margin width 2/||w|| = 2*sqrt(2)", abs(2 / np.linalg.norm(w_feat) - 2 * np.sqrt(2)) < 1e-5)


def c3() -> None:
    print("C3: k(x, z) = ||x - z|| is not positive definite")
    X = np.array([[0.0], [1.0]])
    K = np.abs(X - X.T)
    c = np.array([1.0, -1.0])
    check("two-point counterexample c^T K c = -2", abs(c @ K @ c + 2) < 1e-12, f"K={K.tolist()}, c^T K c={c @ K @ c}")
    check("eigenvalues of that Gram matrix are +-1", np.allclose(np.sort(np.linalg.eigvalsh(K)), [-1, 1]))
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 5))
    G = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    lam_min = np.linalg.eigvalsh(G).min()
    check("random 30-point Gram matrix also has a negative eigenvalue", lam_min < 0, f"lambda_min={lam_min:.3f}")


if __name__ == "__main__":
    c1()
    c2()
    c3()
    print("\nAll checks passed." if not failures else f"\n{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1 if failures else 0)
