"""Lightweight sklearn compatibility base classes.

Allows all estimators in `svmlib` to work seamlessly inside
`sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`
when scikit-learn is installed (e.g., on Google Colab), while remaining
fully functional in pure NumPy/SciPy environments without scikit-learn.
"""

from __future__ import annotations

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError:  # pragma: no cover - fallback when sklearn is not installed
    import inspect

    class BaseEstimator:
        """Minimal fallback replicating sklearn's get_params / set_params."""

        def get_params(self, deep: bool = True) -> dict:
            sig = inspect.signature(self.__init__)
            return {
                name: getattr(self, name)
                for name in sig.parameters
                if name != "self"
            }

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class ClassifierMixin:
        """Minimal fallback replicating sklearn's accuracy score."""

        def score(self, X, y):
            import numpy as np

            return float(np.mean(self.predict(X) == np.asarray(y)))

    class RegressorMixin:
        """Minimal fallback replicating sklearn's R^2 score."""

        def score(self, X, y):
            import numpy as np

            y_true = np.asarray(y, dtype=float)
            y_pred = self.predict(X)
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
