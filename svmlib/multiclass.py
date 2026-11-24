"""Multiclass SVM wrappers: One-vs-One (OvO) and One-vs-Rest (OvR).

- OneVsOneClassifier:
    Trains K(K-1)/2 binary classifiers. Prediction uses majority voting with
    tie-breaking by sum of signed decision values. Probabilities use Platt
    scaling on each pair combined via Wu, Lin & Weng (2004) pairwise coupling.
- OneVsRestClassifier:
    Trains K binary classifiers. Prediction selects argmax decision value.
    Probabilities use per-class Platt scaling normalized to sum to 1.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np

from ._base import BaseEstimator, ClassifierMixin
from .platt import PlattCalibrator, pairwise_coupling


class OneVsOneClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-One multiclass classifier with optional Platt + pairwise coupling."""

    def __init__(
        self,
        estimator_factory: Callable[[], object],
        probability: bool = False,
        cv: int = 5,
        random_state: Optional[int] = 42,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.probability = probability
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsOneClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        if K < 2:
            raise ValueError(f"Need at least 2 classes, got {K}.")

        self.pairs_: List[Tuple[int, int]] = []
        self.estimators_: List[object] = []
        self.calibrators_: List[Optional[PlattCalibrator]] = []
        support_indices = set()

        for i in range(K):
            for j in range(i + 1, K):
                c_i, c_j = self.classes_[i], self.classes_[j]
                mask = (y == c_i) | (y == c_j)
                orig_idx = np.flatnonzero(mask)
                X_pair = X[mask]
                # Map c_i -> +1, c_j -> -1 so positive decision value favors class i
                y_pair = np.where(y[mask] == c_i, 1, -1)

                est = self.estimator_factory()
                est.fit(X_pair, y_pair)
                self.pairs_.append((i, j))
                self.estimators_.append(est)

                if hasattr(est, "support_"):
                    support_indices.update(orig_idx[est.support_].tolist())

                if self.probability:
                    cal = PlattCalibrator(cv=self.cv, random_state=self.random_state)
                    cal.fit(self.estimator_factory, X_pair, y_pair)
                    self.calibrators_.append(cal)
                else:
                    self.calibrators_.append(None)

        self.support_ = np.array(sorted(support_indices), dtype=int)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return pairwise decision values of shape (n_samples, K*(K-1)/2)."""
        X = np.asarray(X, dtype=np.float64)
        cols = [est.decision_function(X) for est in self.estimators_]
        return np.column_stack(cols)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict via majority voting; break ties using sum of signed decision values."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        K = self.classes_.size
        votes = np.zeros((n, K), dtype=np.int64)
        dec_sums = np.zeros((n, K), dtype=np.float64)

        for (i, j), est in zip(self.pairs_, self.estimators_):
            scores = est.decision_function(X)
            # est mapped c_i -> +1, c_j -> -1
            pos = scores >= 0.0
            votes[pos, i] += 1
            votes[~pos, j] += 1
            dec_sums[:, i] += scores
            dec_sums[:, j] -= scores

        # Primary key: votes; tie-breaker: dec_sums
        pred_idx = np.empty(n, dtype=int)
        for s in range(n):
            max_v = votes[s].max()
            tied = np.flatnonzero(votes[s] == max_v)
            if tied.size == 1:
                pred_idx[s] = tied[0]
            else:
                pred_idx[s] = tied[int(np.argmax(dec_sums[s, tied]))]
        return self.classes_[pred_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate multiclass probabilities via pairwise coupling (Wu, Lin & Weng, 2004)."""
        if not self.probability:
            raise RuntimeError("predict_proba requires probability=True.")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        K = self.classes_.size
        R_all = np.zeros((n, K, K), dtype=np.float64)

        for (i, j), est, cal in zip(self.pairs_, self.estimators_, self.calibrators_):
            scores = est.decision_function(X)
            # cal predicts P(y = +1 | f) = P(y = c_i | y in {c_i, c_j})
            r_ij = cal.predict_proba(scores)
            R_all[:, i, j] = r_ij
            R_all[:, j, i] = 1.0 - r_ij

        proba = np.empty((n, K), dtype=np.float64)
        for s in range(n):
            proba[s] = pairwise_coupling(R_all[s])
        return proba


class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-Rest multiclass classifier with optional Platt probability normalization."""

    def __init__(
        self,
        estimator_factory: Callable[[], object],
        probability: bool = False,
        cv: int = 5,
        random_state: Optional[int] = 42,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.probability = probability
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        if K < 2:
            raise ValueError(f"Need at least 2 classes, got {K}.")

        self.estimators_: List[object] = []
        self.calibrators_: List[Optional[PlattCalibrator]] = []
        support_indices = set()

        for c in self.classes_:
            y_bin = np.where(y == c, 1, -1)
            est = self.estimator_factory()
            est.fit(X, y_bin)
            self.estimators_.append(est)
            if hasattr(est, "support_"):
                support_indices.update(est.support_.tolist())

            if self.probability:
                cal = PlattCalibrator(cv=self.cv, random_state=self.random_state)
                cal.fit(self.estimator_factory, X, y_bin)
                self.calibrators_.append(cal)
            else:
                self.calibrators_.append(None)

        self.support_ = np.array(sorted(support_indices), dtype=int)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return decision values of shape (n_samples, K)."""
        X = np.asarray(X, dtype=np.float64)
        return np.column_stack([est.decision_function(X) for est in self.estimators_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class with largest decision value."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict normalized OvR Platt probabilities summing to 1."""
        if not self.probability:
            raise RuntimeError("predict_proba requires probability=True.")
        X = np.asarray(X, dtype=np.float64)
        raw = np.column_stack([
            cal.predict_proba(est.decision_function(X))
            for est, cal in zip(self.estimators_, self.calibrators_)
        ])
        row_sums = raw.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0.0, raw / row_sums, 1.0 / self.classes_.size)
