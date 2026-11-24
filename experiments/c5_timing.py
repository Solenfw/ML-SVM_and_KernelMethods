"""C5 Experiment: Training time comparison between linear SMO and DCD (n = 10^3 .. 10^5).

Specifications (T2.6):
- Synthetic linear dataset with d = 20 features
- n in {1_000, 3_000, 10_000, 30_000, 100_000}
- Same C = 1.0 and tol = 1e-3 for both solvers
- SMO uses on-the-fly kernel columns (cache_kernel=False) so memory stays O(n * d)
- Default timeout 1800s (30 min) per run, 3 repeats averaged (configurable via CLI)

Outputs:
- experiments/c5_timing_results.csv
- report/figures/C5_timing.pdf
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmlib.dcd import LinearSVMDCD
from svmlib.smo import BinarySMO


def make_linear_benchmark_data(n: int, d: int = 20, seed: int = 42):
    """Generate linearly separable data with controlled overlap (margin noise)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    w_true = rng.normal(0.0, 1.0, size=d)
    w_true /= np.linalg.norm(w_true)
    logits = X @ w_true + rng.normal(0.0, 0.35, size=n)
    y = np.where(logits >= 0.0, 1, -1)
    return X, y


def _run_solver_worker(solver_name: str, X: np.ndarray, y: np.ndarray, C: float, tol: float, seed: int, q: mp.Queue):
    t0 = time.perf_counter()
    if solver_name == "SMO":
        model = BinarySMO(
            C=C, kernel="linear", tol=tol, max_iter=500_000, cache_kernel=False
        ).fit(X, y)
    else:
        model = LinearSVMDCD(
            C=C, tol=tol, max_iter=1000, fit_intercept=True, random_state=seed
        ).fit(X, y)
    elapsed = time.perf_counter() - t0
    q.put({
        "time_s": elapsed,
        "n_iter": int(model.n_iter_),
        "dual_obj": float(model.dual_objective_),
        "converged": bool(model.converged_),
    })


def run_with_timeout(solver_name: str, X: np.ndarray, y: np.ndarray, C: float, tol: float, seed: int, timeout_s: float):
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_solver_worker, args=(solver_name, X, y, C, tol, seed, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"time_s": np.nan, "n_iter": -1, "dual_obj": np.nan, "converged": False, "timed_out": True}
    res = q.get() if not q.empty() else {"time_s": np.nan, "n_iter": -1, "dual_obj": np.nan, "converged": False}
    res["timed_out"] = False
    return res


def plot_c5_timing(df: pd.DataFrame, fig_dir: Path, preview_dir: Path | None = None):
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)

    styles = {
        "SMO": dict(color="#d95f02", marker="o", linewidth=2.0, label="SMO tuyến tính (tính cột động)"),
        "DCD": dict(color="#1f5a99", marker="s", linewidth=2.0, label="DCD (Hsieh et al., 2008)"),
    }

    for solver in ("SMO", "DCD"):
        sub = df[(df["solver"] == solver) & (~df["timed_out"])].sort_values("n")
        if len(sub) > 0:
            ax.loglog(sub["n"], sub["mean_time_s"], markersize=6.5, **styles[solver])
            if sub["std_time_s"].notna().any() and (sub["std_time_s"] > 0).any():
                lo = np.maximum(sub["mean_time_s"] - sub["std_time_s"], 1e-4)
                hi = sub["mean_time_s"] + sub["std_time_s"]
                ax.fill_between(sub["n"], lo, hi, color=styles[solver]["color"], alpha=0.15)

        timed = df[(df["solver"] == solver) & (df["timed_out"])].sort_values("n")
        if len(timed) > 0:
            cap = float(timed["timeout_s"].iloc[0])
            ax.scatter(
                timed["n"], [cap] * len(timed),
                marker="x", s=70, color=styles[solver]["color"], zorder=5,
                label=f"{solver} vượt ngưỡng ({int(cap)}s)",
            )

    # Reference O(n) and O(n^2) slopes anchored at n=1000
    n_ref = np.array([1_000, 100_000], dtype=float)
    dcd_1k = df[(df["solver"] == "DCD") & (df["n"] == 1000)]["mean_time_s"]
    if len(dcd_1k) > 0 and np.isfinite(dcd_1k.iloc[0]):
        t0 = float(dcd_1k.iloc[0])
        ax.loglog(n_ref, t0 * (n_ref / 1000.0), ":", color="#555555", linewidth=1.1, label="Độ dốc tham chiếu $O(n)$")
    smo_1k = df[(df["solver"] == "SMO") & (df["n"] == 1000)]["mean_time_s"]
    if len(smo_1k) > 0 and np.isfinite(smo_1k.iloc[0]):
        t0 = float(smo_1k.iloc[0])
        ax.loglog(n_ref, t0 * (n_ref / 1000.0) ** 2, "--", color="#888888", linewidth=1.1, label="Độ dốc tham chiếu $O(n^2)$")

    ax.set_xlabel("Số mẫu huấn luyện $n$ (thang log)")
    ax.set_ylabel("Thời gian huấn luyện (giây, thang log)")
    ax.set_title("C5: So sánh thời gian huấn luyện SMO vs DCD ($d=20$, kernel tuyến tính)")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    fig.tight_layout()

    pdf_path = fig_dir / "C5_timing.pdf"
    fig.savefig(pdf_path)
    if preview_dir is not None:
        fig.savefig(preview_dir / "C5_timing_preview.png", dpi=150)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="C5: Linear SMO vs DCD timing benchmark.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per (solver, n).")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per-run timeout in seconds (default 30 min).")
    parser.add_argument("--tol", type=float, default=1e-3, help="Stopping tolerance.")
    parser.add_argument("--C", type=float, default=1.0, help="Penalty parameter C.")
    parser.add_argument("--preview-dir", type=str, default="", help="Optional directory to save PNG preview.")
    args = parser.parse_args()

    sizes = [1_000, 3_000, 10_000, 30_000, 100_000]
    rows = []
    skip_solver = {"SMO": False, "DCD": False}

    for n in sizes:
        X, y = make_linear_benchmark_data(n=n, d=20, seed=42)
        for solver in ("DCD", "SMO"):
            if skip_solver[solver]:
                rows.append({
                    "n": n, "solver": solver, "mean_time_s": np.nan, "std_time_s": np.nan,
                    "mean_iter": -1, "dual_obj": np.nan, "timed_out": True, "timeout_s": args.timeout,
                })
                print(f"n={n:7d} | {solver:3s} | SKIPPED (previous size timed out)")
                continue

            times, iters, objs = [], [], []
            timed_out = False
            for r in range(args.repeats):
                res = run_with_timeout(solver, X, y, args.C, args.tol, seed=42 + r, timeout_s=args.timeout)
                if res["timed_out"]:
                    timed_out = True
                    skip_solver[solver] = True
                    break
                times.append(res["time_s"])
                iters.append(res["n_iter"])
                objs.append(res["dual_obj"])

            if timed_out:
                rows.append({
                    "n": n, "solver": solver, "mean_time_s": np.nan, "std_time_s": np.nan,
                    "mean_iter": -1, "dual_obj": np.nan, "timed_out": True, "timeout_s": args.timeout,
                })
                print(f"n={n:7d} | {solver:3s} | TIMEOUT (> {args.timeout:.0f}s)")
            else:
                mean_t = float(np.mean(times))
                std_t = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
                mean_it = float(np.mean(iters))
                mean_obj = float(np.mean(objs))
                rows.append({
                    "n": n, "solver": solver, "mean_time_s": mean_t, "std_time_s": std_t,
                    "mean_iter": mean_it, "dual_obj": mean_obj, "timed_out": False, "timeout_s": args.timeout,
                })
                print(f"n={n:7d} | {solver:3s} | time={mean_t:7.3f}s ± {std_t:.3f}s | iter={mean_it:8.1f} | D={mean_obj:.2f}")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "experiments" / "c5_timing_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    plot_c5_timing(df, fig_dir, preview_dir=preview_dir)


if __name__ == "__main__":
    main()
