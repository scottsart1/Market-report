"""Model zoo: transparent baselines through gradient boosting.

All models share the sklearn interface (fit(X, y, sample_weight),
predict_proba(X)) and operate on a feature DataFrame. Scalers and imputers
live inside each model's Pipeline, so in cross-validation they are fit on
training folds only — no scaler ever sees future observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False


class FeatureSubset(BaseEstimator, TransformerMixin):
    """Select named columns from a feature DataFrame (missing ones -> NaN)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.reindex(columns=self.columns).to_numpy(dtype=float)


class ProbitModel(BaseEstimator, ClassifierMixin):
    """Statsmodels probit with median imputation and scaling — the classic
    econometric yield-curve recession model."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.subset_ = FeatureSubset(self.columns).fit(X)
        Z = self.subset_.transform(X)
        self.imputer_ = SimpleImputer(strategy="median").fit(Z)
        Z = self.imputer_.transform(Z)
        self.scaler_ = StandardScaler().fit(Z)
        Z = self.scaler_.transform(Z)
        Z = sm.add_constant(Z, has_constant="add")
        model = sm.Probit(np.asarray(y, dtype=float), Z)
        try:
            self.res_ = model.fit(disp=0, maxiter=200)
        except Exception:
            self.res_ = model.fit(method="bfgs", disp=0, maxiter=500)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        Z = self.scaler_.transform(self.imputer_.transform(self.subset_.transform(X)))
        Z = sm.add_constant(Z, has_constant="add")
        p = np.clip(self.res_.predict(Z), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])


class ConstantModel(BaseEstimator, ClassifierMixin):
    """Baseline: unconditional (weighted) historical event frequency."""

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        self.p_ = float(np.average(np.asarray(y, dtype=float), weights=w))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), np.clip(self.p_, 1e-9, 1 - 1e-9))
        return np.column_stack([1 - p, p])


def _make_logreg(C: float, l1_ratio: float) -> LogisticRegression:
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    kw = dict(solver="saga", C=C, max_iter=8000, tol=1e-5, random_state=0)
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, **kw)
    return LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, **kw)


def _logistic_pipeline(columns: list[str], C: float, l1_ratio: float = 0.5) -> Pipeline:
    return Pipeline([
        ("subset", FeatureSubset(columns)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", _make_logreg(C, l1_ratio)),
    ])


class ElasticNetLogit(BaseEstimator, ClassifierMixin):
    """Elastic-net logistic regression with a small *time-ordered* inner
    validation to choose C (last 20% of the training window — never future
    data relative to the training set)."""

    def __init__(self, columns: list[str], Cs=(0.01, 0.03, 0.1, 0.3), l1_ratio: float = 0.5):
        self.columns = columns
        self.Cs = Cs
        self.l1_ratio = l1_ratio

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y = np.asarray(y, dtype=int)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        split = int(len(y) * 0.8)
        best_C, best_loss = self.Cs[0], np.inf
        if y[:split].sum() >= 5 and y[split:].sum() >= 1:
            for C in self.Cs:
                pipe = _logistic_pipeline(self.columns, C, self.l1_ratio)
                pipe.fit(X.iloc[:split], y[:split], clf__sample_weight=w[:split])
                p = np.clip(pipe.predict_proba(X.iloc[split:])[:, 1], 1e-9, 1 - 1e-9)
                loss = -np.average(
                    y[split:] * np.log(p) + (1 - y[split:]) * np.log(1 - p),
                    weights=w[split:],
                )
                if loss < best_loss:
                    best_loss, best_C = loss, C
        self.C_ = best_C
        self.pipe_ = _logistic_pipeline(self.columns, best_C, self.l1_ratio)
        self.pipe_.fit(X, y, clf__sample_weight=w)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.pipe_.predict_proba(X)

    def coefficients(self) -> pd.Series:
        clf = self.pipe_.named_steps["clf"]
        return pd.Series(clf.coef_[0], index=self.columns)


class HGBModel(BaseEstimator, ClassifierMixin):
    """Histogram gradient boosting (handles NaN natively, no extra deps)."""

    def __init__(self, columns: list[str], random_state: int = 0):
        self.columns = columns
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.subset_ = FeatureSubset(self.columns)
        Z = self.subset_.transform(X)
        # Features with no observations in this training window (series that
        # start later) carry no information and break HGB's binning.
        self.keep_ = ~np.all(np.isnan(Z), axis=0)
        Z = Z[:, self.keep_]
        self.clf_ = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250,
            min_samples_leaf=40, l2_regularization=1.0,
            max_features=0.7, random_state=self.random_state,
            early_stopping=False,
        )
        self.clf_.fit(Z, np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.clf_.predict_proba(self.subset_.transform(X)[:, self.keep_])


class EnsembleModel(BaseEstimator, ClassifierMixin):
    """Probability average of already-constructed member models."""

    def __init__(self, members: list):
        self.members = members

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = [clone(m).fit(X, y, sample_weight=sample_weight) for m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.fitted_], axis=0)
        return np.column_stack([1 - ps, ps])


# ------------------------------------------------------------------ weights

def training_weights(y: np.ndarray, next_onset: pd.Series) -> np.ndarray:
    """Class-balanced weights with event-aware normalization: every recession
    event contributes the same total positive weight, so no single episode
    (e.g. the long 2008 run-up) dominates, and positives as a class carry the
    same total weight as negatives."""
    y = np.asarray(y, dtype=int)
    w = np.ones(len(y), dtype=float)
    pos = y == 1
    if pos.sum() == 0:
        return w
    onset_ids = pd.Series(next_onset).astype("datetime64[ns]")
    counts = onset_ids[pos].groupby(onset_ids[pos]).transform("count")
    n_events = onset_ids[pos].nunique()
    # each event's rows sum to 1, then scale positives to match negatives
    w[pos] = (1.0 / counts.to_numpy()) * ((~pos).sum() / max(n_events, 1))
    # normalize to mean 1 for stable regularization strength
    return w * len(w) / w.sum()


# ------------------------------------------------------------- model builds

MODEL_LABELS = {
    "A_yield_curve": "Model A — Yield-curve probit (econometric baseline)",
    "B_elastic_net": "Model B — Elastic-net logistic (broad features)",
    "C_grad_boost": "Model C — Gradient boosting (nonlinear)",
    "D_ensemble": "Model D — Calibrated ensemble (B + C)",
    "bl_constant": "Baseline — constant historical frequency",
    "bl_sahm": "Baseline — Sahm-rule labor model",
    "bl_nfci": "Baseline — financial-conditions model",
}

YIELD_CURVE_COLS = ["SPREAD_10Y3M__level", "SPREAD_10Y3M__chg12m"]
SAHM_COLS = ["SAHM_RT__level", "UNRATE__chg3m"]
NFCI_COLS = ["NFCI__level", "NFCI__chg3m"]


def build_models(feature_cols: list[str]) -> dict[str, BaseEstimator]:
    def _subset_or_constant(model_cls, wanted: list[str], **kw):
        cols = [c for c in wanted if c in feature_cols]
        # If a model's inputs are entirely unavailable, degrade to the
        # unconditional baseline instead of crashing the whole run.
        return model_cls(columns=cols, **kw) if cols else ConstantModel()

    return {
        "A_yield_curve": _subset_or_constant(ProbitModel, YIELD_CURVE_COLS),
        "B_elastic_net": ElasticNetLogit(columns=feature_cols),
        "C_grad_boost": HGBModel(columns=feature_cols),
        "bl_constant": ConstantModel(),
        "bl_sahm": _subset_or_constant(ElasticNetLogit, SAHM_COLS, Cs=(0.3, 1.0)),
        "bl_nfci": _subset_or_constant(ElasticNetLogit, NFCI_COLS, Cs=(0.3, 1.0)),
    }
