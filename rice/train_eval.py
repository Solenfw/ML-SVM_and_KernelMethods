import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from svmlib import LinearSVMDCD, OneVsOneClassifier, brier_score_multiclass
from svmlib import SVC as SvmlibSVC
from svmlib._base import BaseEstimator, ClassifierMixin

FEAT_CSV = Path("data/features.csv")
RES_DIR = Path("data/results")
FIG_DIR = Path("report/figures")

META_COLS = ["grain_id", "image_id", "session", "variety"]
MORPH_COLS = [
    "area_mm2",
    "perimeter_mm",
    "major_axis_mm",
    "minor_axis_mm",
    "aspect_ratio",
    "circularity",
    "solidity",
]


class LinearSVMDCDMulticlass(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, probability=False, cv=5, random_state=42):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.probability = probability
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        self._ovo = OneVsOneClassifier(
            estimator_factory=lambda: LinearSVMDCD(
                C=self.C, tol=self.tol, max_iter=self.max_iter, random_state=self.random_state
            ),
            probability=self.probability,
            cv=self.cv,
            random_state=self.random_state,
        )
        self._ovo.fit(X, y)
        self.classes_ = self._ovo.classes_
        return self

    def decision_function(self, X):
        return self._ovo.decision_function(X)

    def predict(self, X):
        return self._ovo.predict(X)

    def predict_proba(self, X):
        return self._ovo.predict_proba(X)


def get_models():
    return {
        "SVM_Linear": (
            LinearSVMDCDMulticlass(),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "SVM_Poly": (
            SvmlibSVC(kernel="poly", gamma="scale", tol=1e-2, max_iter=15000),
            {"clf__C": [0.1, 1.0, 10.0], "clf__degree": [2, 3]},
        ),
        "SVM_RBF": (
            SvmlibSVC(kernel="rbf", gamma="scale", tol=1e-2, max_iter=15000),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": list(range(1, 22, 2))},
        ),
        "LogReg": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
    }


def evaluate_nested_logo(X, y, groups, estimator, param_grid):
    outer_cv = LeaveOneGroupOut()
    accs, f1s = [], []
    y_true_all, y_pred_all = [], []
    fold_best_params = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        g_tr = groups.iloc[train_idx]

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        inner_cv = GroupKFold(n_splits=len(np.unique(g_tr)))
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        gs.fit(X_tr, y_tr, groups=g_tr)
        preds = gs.predict(X_te)

        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average="macro"))
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(preds.tolist())
        fold_best_params.append({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "fold_best_params": fold_best_params,
    }


def evaluate_platt_calibration(X, y, groups, base_estimator, fold_best_params):
    outer_cv = LeaveOneGroupOut()
    y_true_all = []
    proba_all = []
    classes = np.array(sorted(y.unique()))

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        params = base_estimator.get_params()
        params.update(fold_best_params[fold_idx])
        params["probability"] = True
        params["cv"] = 3
        params["random_state"] = 42

        clf = base_estimator.__class__(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)

        y_true_all.extend(y_te.tolist())
        proba_all.append(proba)

    y_true_arr = np.array(y_true_all)
    proba_arr = np.vstack(proba_all)
    brier = brier_score_multiclass(y_true_arr, proba_arr, classes)

    pred_idx = np.argmax(proba_arr, axis=1)
    y_pred_arr = classes[pred_idx]
    max_conf = np.max(proba_arr, axis=1)

    rows = []
    for thr in [0.0, 0.5, 0.6, 0.7, 0.8]:
        keep = max_conf >= thr
        coverage = float(np.mean(keep))
        acc = float(accuracy_score(y_true_arr[keep], y_pred_arr[keep])) if np.any(keep) else 0.0
        rows.append(
            {
                "threshold": thr,
                "coverage": round(coverage, 4),
                "accuracy": round(acc, 4),
                "n_kept": int(np.sum(keep)),
            }
        )

    return brier, pd.DataFrame(rows)


def save_eda_plots(df, feature_cols):
    varieties = sorted(df["variety"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_cols = ["area_mm2", "major_axis_mm", "minor_axis_mm", "aspect_ratio"]
    for ax, col in zip(axes.flat, plot_cols):
        data = [df.loc[df["variety"] == v, col].values for v in varieties]
        ax.boxplot(data, tick_labels=varieties)
        ax.set_title(col)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_boxplots.png", dpi=200)
    plt.close(fig)

    X_scaled = StandardScaler().fit_transform(df[feature_cols].values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for v in varieties:
        idx = df["variety"] == v
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.5, label=v)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA 2D - 5 Rice Varieties")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_pca_2d.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    src = "svmlib, LOGO CV" if model_name.startswith("SVM_") else "LOGO CV"
    ax.set_title(f"Confusion Matrix - {model_name} ({src})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEAT_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    color_cols = [c for c in feature_cols if c.startswith(("rgb_", "hsv_", "lab_"))]
    glcm_cols = [c for c in feature_cols if c.startswith("glcm_")]

    save_eda_plots(df, feature_cols)

    X = df[feature_cols]
    y = df["variety"]
    groups = df["session"]
    labels = sorted(y.unique())

    models = get_models()
    summary_rows = []
    best_name = None
    best_f1 = -1.0
    best_res = None
    best_svm_name = None
    best_svm_f1 = -1.0
    best_svm_res = None

    for name, (est, grid) in models.items():
        res = evaluate_nested_logo(X, y, groups, est, grid)
        summary_rows.append(
            {
                "model": name,
                "accuracy_mean": round(res["acc_mean"], 4),
                "accuracy_std": round(res["acc_std"], 4),
                "macro_f1_mean": round(res["f1_mean"], 4),
                "macro_f1_std": round(res["f1_std"], 4),
            }
        )
        print(
            f"{name:12s} | Acc: {res['acc_mean']:.4f} +/- {res['acc_std']:.4f} "
            f"| Macro-F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f}"
        )
        if res["f1_mean"] > best_f1:
            best_f1 = res["f1_mean"]
            best_name = name
            best_res = res
        if name.startswith("SVM_") and res["f1_mean"] > best_svm_f1:
            best_svm_f1 = res["f1_mean"]
            best_svm_name = name
            best_svm_res = res

    pd.DataFrame(summary_rows).to_csv(RES_DIR / "model_comparison.csv", index=False)
    save_confusion_matrix(best_res["y_true"], best_res["y_pred"], labels, best_name)

    ablation_sets = {
        "Morphology": MORPH_COLS,
        "Color": color_cols,
        "GLCM": glcm_cols,
        "Morph+Color": MORPH_COLS + color_cols,
        "Morph+GLCM": MORPH_COLS + glcm_cols,
        "Color+GLCM": color_cols + glcm_cols,
        "All": feature_cols,
    }

    best_est, best_grid = models[best_name]
    ablation_rows = []
    print(f"\nAblation study ({best_name}):")
    for set_name, cols in ablation_sets.items():
        res = evaluate_nested_logo(df[cols], y, groups, best_est, best_grid)
        ablation_rows.append(
            {
                "feature_set": set_name,
                "num_features": len(cols),
                "accuracy_mean": round(res["acc_mean"], 4),
                "accuracy_std": round(res["acc_std"], 4),
                "macro_f1_mean": round(res["f1_mean"], 4),
                "macro_f1_std": round(res["f1_std"], 4),
            }
        )
        print(
            f"{set_name:12s} ({len(cols):2d}) | Acc: {res['acc_mean']:.4f} +/- {res['acc_std']:.4f} "
            f"| Macro-F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f}"
        )

    pd.DataFrame(ablation_rows).to_csv(RES_DIR / "ablation_results.csv", index=False)

    best_svm_est, _ = models[best_svm_name]
    brier, platt_df = evaluate_platt_calibration(
        X, y, groups, best_svm_est, best_svm_res["fold_best_params"]
    )
    platt_df.to_csv(RES_DIR / "platt_thresholds.csv", index=False)
    print(f"\nPlatt Calibration ({best_svm_name}) | Multiclass Brier Score: {brier:.4f}")
    print(platt_df.to_string(index=False))
    print(f"\nDa luu ket qua svmlib vao {RES_DIR} va bieu do vao {FIG_DIR}.")


if __name__ == "__main__":
    main()