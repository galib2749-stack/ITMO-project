"""Classical uplift approaches required by the project spec:

0. random_targeting        -- baseline, not an ML model (random permutation score)
1. response_catboost       -- P(Y=1 | X), demonstrates propensity != uplift
2. t_learner_catboost      -- two independent CatBoost models, tau = mu1 - mu0
3. x_learner_catboost      -- full X-Learner: outcome models -> imputed
                              treatment effects -> effect models -> propensity-
                              weighted blend (Kunzel et al. 2019), not a
                              simplified difference-of-two-models shortcut.

All functions are fit on TRAIN and scored on arbitrary feature matrices, so
the same fitted objects can be reused across validation and holdout without
ever re-fitting on holdout data.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor


def random_targeting_score(n, random_state=0):
    """Not a model: a uniformly random permutation used as the uplift score."""
    rng = np.random.RandomState(random_state)
    return rng.rand(n)


def _default_catboost_classifier(random_state, cat_features=None, **kwargs):
    params = dict(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        cat_features=cat_features,
        allow_writing_files=False,
    )
    params.update(kwargs)
    return CatBoostClassifier(**params)


class ResponseModel:
    """P(Y=1 | X) -- ignores treatment entirely. Used only to demonstrate that
    a plain response/propensity-flavoured model is NOT an uplift model:
    ranking by P(Y=1|X) targets people likely to respond regardless of
    treatment, not people who respond BECAUSE of treatment."""

    def __init__(self, random_state=0, cat_features=None):
        self.random_state = random_state
        self.model = _default_catboost_classifier(random_state, cat_features=cat_features)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_uplift(self, X):
        return self.model.predict_proba(X)[:, 1]


class TLearner:
    """Two independent CatBoost classifiers:
        mu1(x) = P(Y=1 | T=1, X=x)
        mu0(x) = P(Y=1 | T=0, X=x)
        tau(x) = mu1(x) - mu0(x)
    """

    def __init__(self, random_state=0, cat_features=None):
        self.random_state = random_state
        self.model_treat = _default_catboost_classifier(random_state, cat_features=cat_features)
        self.model_ctrl = _default_catboost_classifier(random_state, cat_features=cat_features)

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        self.model_treat.fit(X[treatment == 1], y[treatment == 1])
        self.model_ctrl.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X):
        mu1 = self.model_treat.predict_proba(X)[:, 1]
        mu0 = self.model_ctrl.predict_proba(X)[:, 1]
        return mu1 - mu0


class XLearner:
    """Full X-Learner (Kunzel, Sekhon, Bickel & Yu, 2019 -- "Metalearners for
    estimating heterogeneous treatment effects using machine learning").

    Stage 1 (outcome models, same as T-Learner):
        mu1(x) = P(Y=1 | T=1, X=x)   fit on treated units
        mu0(x) = P(Y=1 | T=0, X=x)   fit on control units

    Stage 2 (imputed treatment effects):
        For treated units i:   D1_i = Y_i - mu0(X_i)      (observed - counterfactual control)
        For control units j:   D0_j = mu1(X_j) - Y_j       (counterfactual treated - observed)

    Stage 3 (effect models, regression on the imputed effects):
        tau1(x) = E[D1 | X=x]   fit on treated units' (X_i, D1_i)
        tau0(x) = E[D0 | X=x]   fit on control units' (X_j, D0_j)

    Stage 4 (propensity-weighted combination):
        g(x) = P(T=1 | X=x)   estimated propensity (falls back to the known/
               constant randomization probability when the design is a known
               RCT, which is the case here -- X5 RetailHero is an A/B test
               with ~50/50 assignment, so g(x) is passed in as a constant
               rather than re-estimated, avoiding an unnecessary and noisy
               propensity model on a known-by-design split)
        tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x)
    """

    def __init__(self, random_state=0, cat_features=None, propensity=None):
        self.random_state = random_state
        self.cat_features = cat_features
        self.propensity = propensity  # None => estimate; float => known constant (RCT)
        self.mu1 = _default_catboost_classifier(random_state, cat_features=cat_features)
        self.mu0 = _default_catboost_classifier(random_state, cat_features=cat_features)
        self.tau1 = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05, loss_function="RMSE",
            random_seed=random_state, verbose=False, cat_features=cat_features,
            allow_writing_files=False,
        )
        self.tau0 = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05, loss_function="RMSE",
            random_seed=random_state, verbose=False, cat_features=cat_features,
            allow_writing_files=False,
        )
        self.propensity_model = None

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y).astype(float)

        X_t, y_t = X[treatment == 1], y[treatment == 1]
        X_c, y_c = X[treatment == 0], y[treatment == 0]

        # Stage 1: outcome models
        self.mu1.fit(X_t, y_t)
        self.mu0.fit(X_c, y_c)

        # Stage 2: imputed treatment effects
        mu0_pred_on_treated = self.mu0.predict_proba(X_t)[:, 1]
        mu1_pred_on_control = self.mu1.predict_proba(X_c)[:, 1]
        d1 = y_t - mu0_pred_on_treated
        d0 = mu1_pred_on_control - y_c

        # Stage 3: effect models
        self.tau1.fit(X_t, d1)
        self.tau0.fit(X_c, d0)

        # Stage 4: propensity
        if self.propensity is None:
            self.propensity_model = _default_catboost_classifier(self.random_state, cat_features=self.cat_features)
            self.propensity_model.fit(X, treatment)
        return self

    def _propensity_scores(self, X):
        if self.propensity is not None:
            return np.full(len(X), float(self.propensity))
        return self.propensity_model.predict_proba(X)[:, 1]

    def predict_uplift(self, X):
        tau1_pred = self.tau1.predict(X)
        tau0_pred = self.tau0.predict(X)
        g = self._propensity_scores(X)
        return g * tau0_pred + (1 - g) * tau1_pred
