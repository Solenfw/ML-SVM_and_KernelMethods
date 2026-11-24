"""A4 Report Table Generator: Detailed comparison of svmlib vs LIBSVM (sklearn.svm.SVC).

Run on Google Colab (where scikit-learn is installed):
    python experiments/a4_report_table.py

Outputs:
    experiments/a4_libsvm_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SkSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmlib import SVC, compute_kernel_matrix


def sklearn_dual_obj(sk: SkSVC, X: np.ndarray, kernel: str, gamma: float, degree: int, coef0: float) -> float:
    sv = sk.support_
    dc = sk.dual_coef_[0]
    K_sv = compute_kernel_matrix(X[sv], kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    return float(np.sum(np.abs(dc)) - 0.5 * (dc @ K_sv @ dc))


def main():
    X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)
    bc = load_breast_cancer()
    X_bc = StandardScaler().fit_transform(bc.data)
    y_bc = bc.target
    iris = load_iris()
    m = iris.target < 2
    X_iris = StandardScaler().fit_transform(iris.data[m])
    y_iris = iris.target[m]
    X_syn, y_syn = make_classification(n_samples=180, n_features=8, n_informative=5, random_state=7)
    X_syn = StandardScaler().fit_transform(X_syn)

    datasets = [
        ("Moons (n=200)", X_moons, y_moons, 1.0),
        ("Breast Cancer (n=569)", X_bc, y_bc, 0.05),
        ("Iris Binary (n=100)", X_iris, y_iris, 0.5),
        ("Synthetic (n=180)", X_syn, y_syn, 0.2),
    ]

    rows = []
    tol = 1e-6
    for ds_name, X, y, gamma in datasets:
        for kernel in ("linear", "poly", "rbf"):
            for C in (0.1, 1.0, 10.0):
                ours = SVC(C=C, kernel=kernel, gamma=gamma, degree=3, coef0=1.0, tol=tol, max_iter=200_000).fit(X, y)
                sk = SkSVC(C=C, kernel=kernel, gamma=gamma, degree=3, coef0=1.0, tol=tol).fit(X, y)

                D_sk = sklearn_dual_obj(sk, X, kernel, gamma, 3, 1.0)
                rel_D = abs(ours.dual_objective_ - D_sk) / max(1.0, abs(D_sk))
                db = abs(ours.b_ - float(sk.intercept_[0]))
                df_max = float(np.max(np.abs(ours.decision_function(X) - sk.decision_function(X))))
                pred_match = float(np.mean(ours.predict(X) == sk.predict(X))) * 100.0

                sv_ours = set(np.flatnonzero(ours.alpha_ > 1e-6 * C).tolist())
                sv_sk = set(sk.support_[np.abs(sk.dual_coef_[0]) > 1e-6 * C].tolist())

                rows.append({
                    "dataset": ds_name,
                    "kernel": kernel,
                    "C": C,
                    "D_svmlib": ours.dual_objective_,
                    "D_libsvm": D_sk,
                    "rel_diff_D": rel_D,
                    "abs_diff_b": db,
                    "max_diff_f": df_max,
                    "n_sv_svmlib": len(sv_ours),
                    "n_sv_libsvm": len(sv_sk),
                    "sv_sym_diff": len(sv_ours ^ sv_sk),
                    "pred_match_pct": pred_match,
                    "smo_iters": ours.n_iter_,
                })

    df = pd.DataFrame(rows)
    out_csv = ROOT / "experiments" / "a4_libsvm_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(df[["dataset", "kernel", "C", "rel_diff_D", "abs_diff_b", "sv_sym_diff", "pred_match_pct"]].to_string(index=False))
    print(f"\n[SAVED] {out_csv}")
    print(f"Max rel_diff_D = {df['rel_diff_D'].max():.2e}, Min pred_match = {df['pred_match_pct'].min():.2f}%")


if __name__ == "__main__":
    main()
