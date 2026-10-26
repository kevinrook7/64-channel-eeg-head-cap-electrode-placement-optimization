"""CSP computed from per-trial covariance matrices (speed-up for channel search).

Same math as mne.decoding.CSP with its defaults (cov_est="concat", reg=None,
component_order="mutual_info", log=True), but it works on precomputed
covariances instead of raw time series:

    C_i       = X_i X_i^T / T                    per-trial covariance (channels x channels)
    C_class   = mean of C_i over that class's trials
    C_a w     = lambda (C_a + C_b) w             generalized eigenproblem
    order     = sort by |lambda - 0.5|, largest first (most class-discriminative)
    feature_j = log(w_j^T C_i w_j)                = log(mean over time of (w_j^T x(t))^2)

The covariance of a channel subset S is just C_i[S][:, S], so each greedy
candidate costs a small matrix slice instead of reprocessing every sample.
"""

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def trial_covariances(X):
    """(trials, channels, times) -> (trials, channels, channels), uncentered like MNE."""
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[2]


def subset_covariances(covs, channels):
    """Keep only `channels` (rows and columns) of every trial covariance."""
    idx = np.asarray(channels)
    return covs[:, idx[:, None], idx[None, :]]


class CovCSP(TransformerMixin, BaseEstimator):
    """Binary CSP on per-trial covariance matrices; outputs log-power features."""

    def __init__(self, n_components=6):
        self.n_components = n_components

    def fit(self, covs, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CovCSP supports exactly two classes")
        c_a = covs[y == self.classes_[0]].mean(axis=0)
        c_b = covs[y == self.classes_[1]].mean(axis=0)
        evals, evecs = scipy.linalg.eigh(c_a, c_a + c_b)
        order = np.argsort(np.abs(evals - 0.5))[::-1]
        self.filters_ = evecs[:, order].T[: self.n_components]
        return self

    def transform(self, covs):
        # w^T C w for every trial and filter
        power = np.einsum("fc,ncd,fd->nf", self.filters_, covs, self.filters_)
        return np.log(power)


def make_cov_csp_decoder(seed=0, n_components=6):
    """Covariance-input twin of baseline.make_csp_decoder()."""
    return Pipeline([
        ("csp", CovCSP(n_components=n_components)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
    ])


if __name__ == "__main__":
    # Check: CovCSP must reproduce MNE's CSP on the same data and folds.
    import mne
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from baseline import N_CSP, SEED, make_csp_decoder, task_window
    from preprocess import prepare_epochs

    mne.set_log_level("WARNING")
    epochs = prepare_epochs(1)
    X = task_window(epochs)
    y = epochs.events[:, 2]
    covs = trial_covariances(X)

    mne_csp = make_csp_decoder().named_steps["csp"].fit(X, y)
    cov_csp = CovCSP(N_CSP).fit(covs, y)
    feat_mne, feat_cov = mne_csp.transform(X), cov_csp.transform(covs)
    print("max |feature difference| vs MNE CSP:", np.abs(feat_mne - feat_cov).max())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k in [8, 63]:
        chans = list(range(k))
        n = min(N_CSP, k)
        a = cross_val_score(make_csp_decoder(n_components=n), X[:, chans], y, cv=cv)
        b = cross_val_score(make_cov_csp_decoder(SEED, n), subset_covariances(covs, chans), y, cv=cv)
        print(f"{k} channels: MNE {a.round(3)}  CovCSP {b.round(3)}  same: {np.array_equal(a, b)}")
