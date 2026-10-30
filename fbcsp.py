"""Filter Bank CSP (Ang et al. 2008) built on the existing CovCSP.

Plain CSP (fast_csp.py) uses one 8-30 Hz band. Different people have their
strongest motor-imagery rhythm in different sub-bands (one person at 10-12 Hz,
another at 20-24 Hz), and one wide band blurs them. FBCSP runs CSP in several
NARROW bands and lets the classifier use whichever bands carry the signal.

Recipe (Ang et al., "Filter Bank Common Spatial Pattern (FBCSP)", IJCNN 2008):
  - 9 band-pass filters, 4 Hz wide, from 4 to 40 Hz: 4-8, 8-12, ..., 36-40.
  - CSP in each band -> log-power features (here: CovCSP from fast_csp.py).
  - Feature selection on the concatenated features, TRAINING DATA ONLY
    (Ang used mutual information; we use SelectKBest(mutual_info_classif)).
  - A linear classifier on the selected features (LR, as elsewhere here).

Design for channel selection: the greedy search slices CHANNELS out of per-trial
covariances. So this works on a LIST of per-band covariance stacks (one
(trials, ch, ch) array per band). band_trial_covariances() builds that list once
per recording; subsetting channels is then a cheap slice per band, exactly like
fast_csp.subset_covariances. This keeps FBCSP a drop-in for the CSP decoder in
select_channels.py / evaluate.py without duplicating the greedy logic.

No new dependency: MNE's filter + scikit-learn's mutual_info_classif.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import N_CSP, SEED
from fast_csp import CovCSP, trial_covariances

# Ang et al.: nine 4 Hz bands from 4 to 40 Hz.
FILTER_BANK = [(lo, lo + 4) for lo in range(4, 40, 4)]  # (4,8),(8,12),...,(36,40)
N_FB_FEATURES = 8  # best features kept after mutual-information selection


def band_filter(X, sfreq, l_freq, h_freq):
    """Band-pass raw epoch data (trials, channels, times) with MNE's FIR filter."""
    import mne

    return mne.filter.filter_data(
        X, sfreq, l_freq, h_freq, verbose=False, phase="zero", fir_design="firwin"
    )


def band_trial_covariances(X, sfreq, bands=FILTER_BANK):
    """List of per-band per-trial covariance stacks: one (trials, ch, ch) per band.

    Filters one band at a time and frees the filtered copy before the next, so
    peak memory holds only one band of (trials, channels, times) at once.
    """
    covs = []
    for l_freq, h_freq in bands:
        Xb = band_filter(X, sfreq, l_freq, h_freq)
        covs.append(trial_covariances(Xb))
        del Xb
    return covs


def subset_band_covariances(band_covs, channels):
    """Slice `channels` (rows+cols) out of every band's covariances."""
    idx = np.asarray(channels)
    return [c[:, idx[:, None], idx[None, :]] for c in band_covs]


class FBCSP(BaseEstimator, ClassifierMixin):
    """Filter Bank CSP + MI feature selection + logistic regression.

    Input X is a LIST of per-band covariance stacks (output of
    band_trial_covariances or subset_band_covariances), NOT raw time series.
    Everything is fit on training trials only, so it is leakage-safe inside CV.
    """

    def __init__(self, n_components=N_CSP, k_features=N_FB_FEATURES, seed=SEED):
        self.n_components = n_components
        self.k_features = k_features
        self.seed = seed

    def _features(self, band_covs, fit):
        feats = []
        if fit:
            self.csps_ = []
        for b, covs in enumerate(band_covs):
            n_comp = min(self.n_components, covs.shape[1])
            if fit:
                csp = CovCSP(n_components=n_comp).fit(covs, self._y)
                self.csps_.append(csp)
            feats.append(self.csps_[b].transform(covs))
        return np.concatenate(feats, axis=1)

    def fit(self, band_covs, y):
        self._y = y
        X = self._features(band_covs, fit=True)
        k = min(self.k_features, X.shape[1])
        self.select_ = SelectKBest(
            lambda Xf, yf: mutual_info_classif(Xf, yf, random_state=self.seed), k=k
        ).fit(X, y)
        self.clf_ = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)),
        ]).fit(self.select_.transform(X), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, band_covs):
        X = self._features(band_covs, fit=False)
        return self.clf_.predict(self.select_.transform(X))

    def score(self, band_covs, y):
        return float(np.mean(self.predict(band_covs) == y))


if __name__ == "__main__":
    # Self-test on subject 1: FBCSP should at least match plain CSP within-session.
    import time

    import mne
    from sklearn.model_selection import StratifiedKFold

    from baseline import task_window
    from fast_csp import make_cov_csp_decoder, subset_covariances
    from load_bnci import prepare_bnci_sessions

    mne.set_log_level("WARNING")
    print("bands:", FILTER_BANK)

    sessions = prepare_bnci_sessions(1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, epochs in sessions.items():
        X = task_window(epochs)
        y = epochs.events[:, 2]
        sfreq = epochs.info["sfreq"]

        start = time.time()
        band_covs = band_trial_covariances(X, sfreq)
        wide_covs = trial_covariances(X)

        fb_scores, csp_scores = [], []
        for tr, te in cv.split(np.zeros(len(y)), y):
            fb = FBCSP().fit([c[tr] for c in band_covs], y[tr])
            fb_scores.append(fb.score([c[te] for c in band_covs], y[te]))

            csp = make_cov_csp_decoder(SEED).fit(wide_covs[tr], y[tr])
            csp_scores.append(csp.score(wide_covs[te], y[te]))

        fb_scores, csp_scores = np.array(fb_scores), np.array(csp_scores)
        print(f"session {name}: FBCSP {fb_scores.mean():.3f} +/- {fb_scores.std(ddof=1):.3f}"
              f"   plain CSP {csp_scores.mean():.3f} +/- {csp_scores.std(ddof=1):.3f}"
              f"   ({time.time() - start:.0f}s)")
