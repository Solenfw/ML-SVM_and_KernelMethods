"""C4 Experiment: effect of C and gamma on the RBF SVM (svmlib SMO), and the LOO bound of B6.

For every (C, gamma) on a 6 x 5 grid and two datasets (moons n=200, breast cancer n=569):
- fit on the full (standardized) data: number of SVs (bounded alpha = C / free 0 < alpha < C),
  margin width 2 / ||w|| with ||w||^2 = alpha^T Q alpha, training error;
- 5-fold CV error (StandardScaler inside the pipeline);
- leave-one-out error, compared with the bound  LOO <= n_SV / n  (Theorem B6).
Nested CV: outer 5-fold for an honest test error, inner 5-fold grid search over (C, gamma);
the heatmap shows the inner-CV error averaged over the outer folds.
Two extra hard-margin cases (C = 1e6) match the exact assumptions of B6:
linearly separable blobs with the linear kernel, and moons with the RBF kernel.

Outputs:
- experiments/c4_grid_results.csv, experiments/c4_nested_results.csv, experiments/c4_hard_margin.csv
- report/figures/C4_heatmap.pdf, C4_sv_margin.pdf, C4_boundaries.pdf, C4_loo_bound.pdf

Usage: python experiments/c4_C_gamma.py            (full run, a few minutes)
       python experiments/c4_C_gamma.py --plot-only (redraw figures from the CSVs)
       python experiments/c4_C_gamma.py --jobs 1    (no multiprocessing)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from sklearn.datasets import load_breast_cancer, make_blobs, make_moons
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmlib import SVC

EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "report" / "figures"
GRID_CSV = EXP_DIR / "c4_grid_results.csv"
NESTED_CSV = EXP_DIR / "c4_nested_results.csv"
HARD_CSV = EXP_DIR / "c4_hard_margin.csv"

CS = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
GAMMAS = [1e-3, 1e-2, 1e-1, 1.0, 1e1]
HARD_C = 1e6
SEED = 0
SVM_KW = dict(kernel="rbf", tol=1e-3, max_iter=2_000_000)
N_JOBS = -1  # set by --jobs

DATASETS = {"moons": "Moons (n = 200, d = 2)", "cancer": "Breast cancer (n = 569, d = 30)"}

# Reference palette (dataviz skill, light mode): chrome, categorical slots, blue sequential ramp.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


# ---------------------------------------------------------------------------- data
def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw features and labels in {-1, +1}."""
    if name == "moons":
        X, y = make_moons(n_samples=200, noise=0.25, random_state=SEED)
    else:
        X, y = load_breast_cancer(return_X_y=True)
    return X.astype(float), np.where(y == 1, 1, -1)


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------- one config
def fit_stats(X: np.ndarray, y: np.ndarray, C: float, gamma: float, kernel: str = "rbf") -> dict:
    """Full-data fit: SV counts, margin width, training error, convergence."""
    m = SVC(C=C, gamma=gamma, **{**SVM_KW, "kernel": kernel}).fit(X, y)
    a_sv = m.alpha_[m.support_]
    n_bounded = int(np.sum(a_sv >= C * (1 - 1e-6)))
    sv = m.support_vectors_
    K = linear_kernel(sv) if kernel == "linear" else rbf_kernel(sv, gamma=gamma)
    dc = m.dual_coef_[0]
    w_norm = float(np.sqrt(max(dc @ K @ dc, 1e-300)))
    return {
        "n_sv": len(m.support_),
        "n_bounded": n_bounded,
        "n_free": len(m.support_) - n_bounded,
        "margin_width": 2.0 / w_norm,
        "train_err": float(np.mean(m.predict(X) != y)),
        "converged": bool(m.converged_),
    }


def loo_error(X: np.ndarray, y: np.ndarray, C: float, gamma: float, kernel: str = "rbf") -> float:
    n, wrong = len(y), 0
    for i in range(n):
        mask = np.arange(n) != i
        m = SVC(C=C, gamma=gamma, **{**SVM_KW, "kernel": kernel}).fit(X[mask], y[mask])
        wrong += int(m.predict(X[i : i + 1])[0] != y[i])
    return wrong / n


def pipeline(C: float = 1.0, gamma: float = 1.0) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(C=C, gamma=gamma, **SVM_KW))])


def cv_error(X_raw: np.ndarray, y: np.ndarray, C: float, gamma: float) -> float:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(1 - cross_val_score(pipeline(C, gamma), X_raw, y, cv=cv).mean())


def grid_task(name: str, X_raw, X, y, C, gamma) -> dict:
    row = {"dataset": name, "C": C, "gamma": gamma}
    row.update(fit_stats(X, y, C, gamma))
    row["cv_err"] = cv_error(X_raw, y, C, gamma)
    row["loo_err"] = loo_error(X, y, C, gamma)
    row["sv_frac"] = row["n_sv"] / len(y)
    return row


# ---------------------------------------------------------------------------- nested CV
def nested_cv(name: str, X_raw: np.ndarray, y: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Returns per-outer-fold rows and the inner-CV error grid averaged over outer folds (|C| x |gamma|)."""
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    grid = {"svm__C": CS, "svm__gamma": GAMMAS}
    rows, inner_grids = [], []
    for k, (tr, te) in enumerate(outer.split(X_raw, y)):
        gs = GridSearchCV(pipeline(), grid, cv=inner, n_jobs=N_JOBS).fit(X_raw[tr], y[tr])
        err = np.empty((len(CS), len(GAMMAS)))
        for params, score in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
            err[CS.index(params["svm__C"]), GAMMAS.index(params["svm__gamma"])] = 1 - score
        inner_grids.append(err)
        rows.append({
            "dataset": name, "outer_fold": k,
            "best_C": gs.best_params_["svm__C"], "best_gamma": gs.best_params_["svm__gamma"],
            "inner_cv_err": 1 - gs.best_score_,
            "outer_test_err": float(np.mean(gs.predict(X_raw[te]) != y[te])),
        })
    return rows, np.mean(inner_grids, axis=0)


# ---------------------------------------------------------------------------- hard margin (exact B6 setting)
def hard_margin_cases() -> list[dict]:
    Xb, yb = make_blobs(n_samples=100, centers=[(-2, -2), (2, 2)], cluster_std=0.9, random_state=SEED)
    yb = np.where(yb == 1, 1, -1)
    Xm, ym = load("moons")
    # gamma = 3 for moons: with gamma = 1 two nearly coincident points of opposite classes force alpha = C
    # even at C = 1e6 (LIBSVM agrees), so that fit is still a soft margin, not the hard margin of B6.
    cases = [("blobs, linear", standardize(Xb), yb, "linear", 1.0),
             ("moons, rbf gamma=3", standardize(Xm), ym, "rbf", 3.0)]
    rows = []
    for label, X, y, kernel, gamma in cases:
        st = fit_stats(X, y, HARD_C, gamma, kernel)
        # Hard margin <=> every training point is classified with margin and no alpha reaches C.
        if st["train_err"] > 0 or st["n_bounded"] > 0:
            print(f"WARNING [{label}]: not a hard-margin solution (train_err={st['train_err']}, "
                  f"bounded SVs={st['n_bounded']})")
        rows.append({"case": label, "kernel": kernel, "gamma": gamma, "C": HARD_C, "n": len(y), **st,
                     "sv_frac": st["n_sv"] / len(y), "loo_err": loo_error(X, y, HARD_C, gamma, kernel)})
    return rows


# ---------------------------------------------------------------------------- csv helpers
def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    def conv(v: str):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    with path.open() as fh:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]


# ---------------------------------------------------------------------------- plotting
def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": AXIS_C, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False,
    })


def tick_label(v: float) -> str:
    e = int(round(np.log10(v)))
    return f"$10^{{{e}}}$" if e not in (0, 1) else ("1" if e == 0 else "10")


def plot_heatmap(grid_err: dict, nested: list[dict]) -> None:
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), constrained_layout=True)
    for ax, name in zip(axes, DATASETS):
        E = 100 * grid_err[name]
        im = ax.imshow(E, origin="lower", cmap=cmap, vmin=0, vmax=50, aspect="auto")
        for i in range(len(CS)):
            for j in range(len(GAMMAS)):
                ax.text(j, i, f"{E[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if E[i, j] > 22 else INK)
        bi, bj = np.unravel_index(np.argmin(E), E.shape)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
        rows = [r for r in nested if r["dataset"] == name]
        outer = 100 * np.array([r["outer_test_err"] for r in rows])
        ax.set_title(f"{DATASETS[name]}\nsai số ngoài: {outer.mean():.1f} ± {outer.std(ddof=1):.1f} %", loc="left")
        ax.set_xticks(range(len(GAMMAS)), [tick_label(g) for g in GAMMAS])
        ax.set_yticks(range(len(CS)), [tick_label(c) for c in CS])
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$C$")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("sai số CV vòng trong (%)")
    cb.outline.set_visible(False)
    fig.savefig(FIG_DIR / "C4_heatmap.pdf")
    plt.close(fig)


def series(rows: list[dict], name: str, gamma: float, key: str) -> np.ndarray:
    return np.array([next(r[key] for r in rows if r["dataset"] == name and r["C"] == c and r["gamma"] == gamma) for c in CS])


def plot_sv_margin(rows: list[dict], gamma_show: dict, gamma_star: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), constrained_layout=True)
    for r, name in enumerate(DATASETS):
        ax_sv, ax_m, ax_e = axes[r]
        for k, g in enumerate(gamma_show[name]):
            lab = rf"$\gamma = {tick_label(g).strip('$')}$"
            sv = 100 * series(rows, name, g, "sv_frac")
            mw = series(rows, name, g, "margin_width")
            ax_sv.plot(CS, sv, marker="o", color=CAT[k], label=lab)
            ax_m.plot(CS, mw, marker="o", color=CAT[k], label=lab)
            ax_m.annotate(lab, (CS[-1], mw[-1]), xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=6.5, color=INK_2)
        ax_sv.legend(loc="upper right", handlelength=1.4)
        ax_sv.set_ylabel(f"{DATASETS[name].split(' (')[0]}\n" + r"$n_{\mathrm{SV}}/n$ (%)")
        ax_sv.set_ylim(0, 105)
        ax_m.set_yscale("log")
        ax_m.set_ylabel(r"độ rộng lề $2/\|\mathbf{w}\|$")
        gs = gamma_star[name]
        tr = 100 * series(rows, name, gs, "train_err")
        cv = 100 * series(rows, name, gs, "cv_err")
        ax_e.plot(CS, tr, marker="o", color=CAT[0], label="huấn luyện")
        ax_e.plot(CS, cv, marker="s", color=CAT[1], label="CV 5-fold")
        ax_e.legend(loc="lower left" if name == "moons" else "center right", handlelength=1.4,
                    title=rf"$\gamma = {tick_label(gs).strip('$')}$", title_fontsize=7)
        ax_e.set_ylabel("sai số (%)")
        ax_e.set_ylim(bottom=0)
        for ax in (ax_sv, ax_m, ax_e):
            ax.set_xscale("log")
            ax.set_xticks(CS, [tick_label(c) for c in CS])
            ax.minorticks_off()
            if r == 1:
                ax.set_xlabel(r"$C$")
        ax_m.set_xlim(right=CS[-1] * 12)
    for ax, t in zip(axes[0], ["(a) tỉ lệ vector hỗ trợ", "(b) độ rộng lề", "(c) sai số theo $C$"]):
        ax.set_title(t, loc="left")
    fig.savefig(FIG_DIR / "C4_sv_margin.pdf")
    plt.close(fig)


def plot_boundaries(rows: list[dict]) -> None:
    X_raw, y = load("moons")
    X = standardize(X_raw)
    c_show, g_show = [1e-1, 1.0, 1e2], [1e-1, 1.0, 1e1]
    pad = 0.4
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.0), constrained_layout=True, sharex=True, sharey=True)
    for i, C in enumerate(c_show):
        for j, g in enumerate(g_show):
            ax = axes[i, j]
            m = SVC(C=C, gamma=g, **SVM_KW).fit(X, y)
            F = m.decision_function(grid).reshape(xx.shape)
            ax.contourf(xx, yy, F, levels=[-1e9, 0, 1e9], colors=[CAT[1], CAT[0]], alpha=0.07)
            ax.contour(xx, yy, F, levels=[0], colors=INK, linewidths=1.2)
            ax.contour(xx, yy, F, levels=[-1, 1], colors=MUTED, linewidths=0.9, linestyles="--")
            for cls, col in ((1, CAT[0]), (-1, CAT[1])):
                pts = X[y == cls]
                ax.scatter(pts[:, 0], pts[:, 1], s=9, color=col, edgecolors=SURFACE, linewidths=0.4, zorder=3)
            sv = m.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=26, facecolors="none", edgecolors=INK, linewidths=0.6, zorder=4)
            r = next(r for r in rows if r["dataset"] == "moons" and r["C"] == C and r["gamma"] == g)
            ax.set_title(rf"$C = {tick_label(C).strip('$')},\ \gamma = {tick_label(g).strip('$')}$"
                         + f"\n$n_{{SV}}$ = {int(r['n_sv'])}, CV {100 * r['cv_err']:.1f} %", loc="left", fontsize=7.5)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(AXIS_C)
    handles = [Line2D([], [], color=INK, lw=1.2, label="$f(x) = 0$"),
               Line2D([], [], color=MUTED, lw=0.9, ls="--", label=r"$f(x) = \pm 1$ (biên lề)"),
               Line2D([], [], marker="o", ls="", color=CAT[0], label="lớp $+1$"),
               Line2D([], [], marker="o", ls="", color=CAT[1], label="lớp $-1$"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="vector hỗ trợ")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=7)
    fig.savefig(FIG_DIR / "C4_boundaries.pdf")
    plt.close(fig)


def plot_loo(rows: list[dict], hard: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    ax.fill_between([0, 100], [0, 100], [100, 100], color=GRID_C, alpha=0.35, lw=0)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--")
    ax.text(30, 82, "vùng vi phạm cận\n(LOO > $n_{SV}/n$)", color=INK_2, fontsize=7, ha="center")
    for k, name in enumerate(DATASETS):
        pts = [r for r in rows if r["dataset"] == name]
        ax.scatter([100 * r["sv_frac"] for r in pts], [100 * r["loo_err"] for r in pts], s=22, color=CAT[k],
                   edgecolors=SURFACE, linewidths=0.8, label=f"{DATASETS[name].split(' (')[0]} (30 cấu hình)",
                   zorder=3, clip_on=False)
    ax.scatter([100 * h["sv_frac"] for h in hard], [100 * h["loo_err"] for h in hard], s=40, marker="D",
               color=CAT[2], edgecolors=SURFACE, linewidths=0.8, label=r"lề cứng ($C = 10^6$)", zorder=4, clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"tỉ lệ vector hỗ trợ $n_{\mathrm{SV}}/n$ (%)")
    ax.set_ylabel("sai số bỏ-một (LOO, %)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), handletextpad=0.3)
    fig.savefig(FIG_DIR / "C4_loo_bound.pdf")
    plt.close(fig)


def make_figures() -> None:
    rows, nested, hard = read_csv(GRID_CSV), read_csv(NESTED_CSV), read_csv(HARD_CSV)
    grid_err = {}
    for name in DATASETS:
        E = np.empty((len(CS), len(GAMMAS)))
        for r in rows:
            if r["dataset"] == name:
                E[CS.index(r["C"]), GAMMAS.index(r["gamma"])] = r["nested_inner_err"]
        grid_err[name] = E
    # gamma most often chosen by the inner search (ties -> smaller gamma) drives panel (c)
    gamma_star = {}
    for name in DATASETS:
        chosen = [r["best_gamma"] for r in nested if r["dataset"] == name]
        gamma_star[name] = min(set(chosen), key=lambda g: (-chosen.count(g), g))
    gamma_show = {"moons": [1e-1, 1.0, 1e1], "cancer": [1e-3, 1e-2, 1e-1]}
    style()
    plot_heatmap(grid_err, nested)
    plot_sv_margin(rows, gamma_show, gamma_star)
    plot_boundaries(rows)
    plot_loo(rows, hard)
    print(f"Figures written to {FIG_DIR.relative_to(ROOT)}/C4_*.pdf  (gamma* = {gamma_star})")


# ---------------------------------------------------------------------------- main
def run() -> None:
    t0 = time.perf_counter()
    all_rows, all_nested = [], []
    for name in DATASETS:
        X_raw, y = load(name)
        X = standardize(X_raw)
        rows = Parallel(n_jobs=N_JOBS)(delayed(grid_task)(name, X_raw, X, y, C, g) for C in CS for g in GAMMAS)
        nested, inner_grid = nested_cv(name, X_raw, y)
        for r in rows:
            r["nested_inner_err"] = float(inner_grid[CS.index(r["C"]), GAMMAS.index(r["gamma"])])
        all_rows += rows
        all_nested += nested
        write_csv(GRID_CSV, all_rows)
        write_csv(NESTED_CSV, all_nested)
        print(f"[{name}] grid + LOO + nested CV done ({time.perf_counter() - t0:.0f}s)")
    hard = hard_margin_cases()
    write_csv(HARD_CSV, hard)
    print(f"hard-margin cases done ({time.perf_counter() - t0:.0f}s)")

    bad = [r for r in all_rows if not r["converged"]]
    print(f"non-converged fits on the full grid: {len(bad)}")
    viol = [r for r in all_rows + hard if r["loo_err"] > r["sv_frac"] + 1e-12]
    print(f"LOO bound violations: {len(viol)} / {len(all_rows) + len(hard)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    args = ap.parse_args()
    N_JOBS = args.jobs
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run()
    make_figures()
