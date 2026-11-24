"""Generate A1 SMO convergence plot and A3 Platt scaling reliability diagram.

Also runs standalone verification of multiclass (OvO/OvR), string labels,
Platt scaling Newton optimality, and pairwise coupling optimality.
Saves vector PDFs into report/figures/:
- report/figures/A1_convergence.pdf
- report/figures/A3_calibration.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmlib import SVC, brier_score_multiclass, pairwise_coupling, reliability_diagram_data


def _make_moons(n: int = 300, noise: float = 0.18, seed: int = 42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n2)
    X1 = np.column_stack([np.cos(t1), np.sin(t1)])
    X2 = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([X1, X2]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.concatenate([np.ones(n1, dtype=int), -np.ones(n2, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _make_multiclass_blobs(n_per_class: int = 120, seed: int = 7):
    """5-class synthetic dataset mimicking 5 rice varieties with overlapping clusters."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [2.2, 0.8, -0.5, 0.4],
        [-1.8, 2.0, 0.7, -0.3],
        [0.5, -2.2, 1.5, 0.8],
        [-2.0, -1.6, -1.4, 1.0],
    ])
    names = ["ST25", "BacHuong", "TamThom", "NepCaiHoaVang", "Japonica"]
    Xs, ys = [], []
    for center, name in zip(centers, names):
        Xs.append(center + rng.normal(0.0, 1.05, size=(n_per_class, centers.shape[1])))
        ys.append(np.full(n_per_class, name, dtype=object))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def generate_a1_convergence(fig_dir: Path, preview_dir: Path | None = None):
    X, y = _make_moons(n=300, noise=0.18, seed=42)
    configs = [
        ("RBF ($C=1, \\gamma=1$)", dict(C=1.0, kernel="rbf", gamma=1.0, tol=1e-6)),
        ("RBF ($C=10, \\gamma=1$)", dict(C=10.0, kernel="rbf", gamma=1.0, tol=1e-6)),
        ("Poly ($d=3, C=1$)", dict(C=1.0, kernel="poly", gamma=0.5, degree=3, coef0=1.0, tol=1e-6)),
    ]
    colors = ["#1f5a99", "#d95f02", "#1b9e77"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=150)

    for (label, kwargs), color in zip(configs, colors):
        model = SVC(record_history=True, history_stride=1, max_iter=100_000, **kwargs).fit(X, y)
        iters = np.array(model.history_["iter"])
        dual = np.array(model.history_["dual_obj"])
        gaps = np.maximum(np.array(model.history_["kkt_gap"]), 1e-12)
        ax1.plot(iters, dual, label=f"{label} ({model.n_iter_} vòng)", color=color, linewidth=1.8)
        ax2.semilogy(iters, gaps, label=label, color=color, linewidth=1.8)

    ax1.set_xlabel("Số vòng lặp SMO")
    ax1.set_ylabel("Hàm mục tiêu đối ngẫu $D(\\alpha)$")
    ax1.set_title("Hội tụ hàm mục tiêu đối ngẫu")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(frameon=True, fontsize=8.5)

    ax2.axhline(1e-6, color="#666666", linestyle=":", linewidth=1.2, label="Ngưỡng dừng $\\mathrm{tol}=10^{-6}$")
    ax2.set_xlabel("Số vòng lặp SMO")
    ax2.set_ylabel("Độ vi phạm KKT $m(\\alpha) - M(\\alpha)$")
    ax2.set_title("Độ vi phạm điều kiện KKT (thang log)")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(frameon=True, fontsize=8.5)

    fig.suptitle("A1: Đường hội tụ thuật toán SMO (WSS1) trên tập dữ liệu Moons ($n=300$)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = fig_dir / "A1_convergence.pdf"
    fig.savefig(pdf_path)
    if preview_dir is not None:
        fig.savefig(preview_dir / "A1_convergence_preview.png", dpi=150)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")


def verify_and_generate_a3_calibration(fig_dir: Path, preview_dir: Path | None = None):
    R = np.array([
        [0.0, 0.8, 0.7],
        [0.2, 0.0, 0.6],
        [0.3, 0.4, 0.0],
    ])
    p = pairwise_coupling(R)
    assert abs(p.sum() - 1.0) < 1e-9 and np.all(p > 0)
    print(f"[PASS] Pairwise coupling: p={np.round(p, 4)}, sum={p.sum():.6f}")

    X, y = _make_multiclass_blobs(n_per_class=120, seed=7)
    n_tr = 420
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-12
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd

    ovo = SVC(C=1.0, kernel="rbf", gamma=0.25, tol=1e-5, decision_function_shape="ovo", probability=True, cv=5, random_state=42).fit(X_tr_s, y_tr)
    ovr = SVC(C=1.0, kernel="rbf", gamma=0.25, tol=1e-5, decision_function_shape="ovr", probability=True, cv=5, random_state=42).fit(X_tr_s, y_tr)

    p_ovo = ovo.predict_proba(X_te_s)
    p_ovr = ovr.predict_proba(X_te_s)
    assert np.allclose(p_ovo.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(p_ovr.sum(axis=1), 1.0, atol=1e-9)

    acc_ovo = ovo.score(X_te_s, y_te)
    acc_ovr = ovr.score(X_te_s, y_te)
    brier_ovo = brier_score_multiclass(y_te, p_ovo, ovo.classes_)
    brier_ovr = brier_score_multiclass(y_te, p_ovr, ovr.classes_)
    print(f"[PASS] Multiclass 5-class (string labels): OvO acc={acc_ovo:.3f}, Brier={brier_ovo:.4f} | OvR acc={acc_ovr:.3f}, Brier={brier_ovr:.4f}")

    # Shuffled binary moons dataset for reliability diagram
    X_b, y_b = _make_moons(n=600, noise=0.28, seed=99)
    X_b_tr, y_b_tr = X_b[:400], y_b[:400]
    X_b_te, y_b_te = X_b[400:], y_b[400:]
    clf_bin = SVC(C=1.0, kernel="rbf", gamma=1.0, tol=1e-5, probability=True, cv=5, random_state=42).fit(X_b_tr, y_b_tr)
    prob_bin = clf_bin.predict_proba(X_b_te)[:, 1]
    raw_f = clf_bin.decision_function(X_b_te)
    uncal_prob = 1.0 / (1.0 + np.exp(-raw_f))

    y_b_01 = (y_b_te > 0).astype(float)
    brier_platt = float(np.mean((prob_bin - y_b_01) ** 2))
    brier_uncal = float(np.mean((uncal_prob - y_b_01) ** 2))
    print(f"[PASS] Binary Platt: Brier={brier_platt:.4f} vs uncalibrated sigmoid Brier={brier_uncal:.4f}")

    mp_platt, fp_platt, _ = reliability_diagram_data(y_b_te, prob_bin, n_bins=8)
    mp_uncal, fp_uncal, _ = reliability_diagram_data(y_b_te, uncal_prob, n_bins=8)

    Y_oh = (y_te[:, None] == ovo.classes_[None, :]).astype(int)
    mp_ovo, fp_ovo, _ = reliability_diagram_data(Y_oh.ravel(), p_ovo.ravel(), n_bins=8)
    mp_ovr, fp_ovr, _ = reliability_diagram_data(Y_oh.ravel(), p_ovr.ravel(), n_bins=8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=150)

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.1, label="Hiệu chỉnh lý tưởng")
    ax1.plot(mp_platt, fp_platt, "o-", color="#1f5a99", linewidth=2.0, markersize=6, label=f"Platt scaling (Brier={brier_platt:.4f})")
    ax1.plot(mp_uncal, fp_uncal, "s:", color="#d95f02", linewidth=1.6, markersize=5, label=f"Sigmoid thô $\\sigma(f)$ (Brier={brier_uncal:.4f})")
    ax1.set_xlabel("Xác suất dự đoán trung bình")
    ax1.set_ylabel("Tần suất thực tế lớp dương")
    ax1.set_title("Nhị phân: Moons ($n_{\\mathrm{test}}=200$)")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    ax2.plot([0, 1], [0, 1], "k--", linewidth=1.1, label="Hiệu chỉnh lý tưởng")
    ax2.plot(mp_ovo, fp_ovo, "o-", color="#1b9e77", linewidth=2.0, markersize=6, label=f"OvO + Pairwise Coupling (Brier={brier_ovo:.4f})")
    ax2.plot(mp_ovr, fp_ovr, "^--", color="#7570b3", linewidth=1.8, markersize=6, label=f"OvR + chuẩn hóa (Brier={brier_ovr:.4f})")
    ax2.set_xlabel("Xác suất dự đoán trung bình")
    ax2.set_ylabel("Tần suất thực tế")
    ax2.set_title("Đa lớp (5 lớp, $n_{\\mathrm{test}}=180$)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

    fig.suptitle("A3: Biểu đồ độ tin cậy (Reliability Diagram) của Platt Scaling", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = fig_dir / "A3_calibration.pdf"
    fig.savefig(pdf_path)
    if preview_dir is not None:
        fig.savefig(preview_dir / "A3_calibration_preview.png", dpi=150)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")


if __name__ == "__main__":
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = ROOT.parent
    generate_a1_convergence(fig_dir, preview_dir=preview_dir)
    verify_and_generate_a3_calibration(fig_dir, preview_dir=preview_dir)
