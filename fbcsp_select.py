"""Greedy channel selection with FBCSP (Filter Bank CSP) as the decoder.

These are the FBCSP twins of select_channels.subset_accuracy / greedy_forward /
test_accuracy. Same algorithm, same nested CV, same leakage rules -- the only
difference is the input. Plain CSP works on ONE covariance array (trials, ch, ch);
FBCSP works on a LIST of per-band covariance stacks (fbcsp.band_trial_covariances),
and subsetting channels is a slice of every band (fbcsp.subset_band_covariances).

Kept separate from select_channels.py on purpose, so the plain-CSP PhysioNet path
is untouched. penalized_size is reused unchanged (it only reads the greedy path).
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from baseline import SEED
from fbcsp import FBCSP, subset_band_covariances
from select_channels import INNER_SPLITS, penalized_size  # reuse unchanged

__all__ = ["fb_subset_accuracy", "fb_greedy_forward", "fb_test_accuracy", "penalized_size"]


def fb_subset_accuracy(band_covs, y, channels, seed=SEED):
    """Inner-CV accuracy of FBCSP using only `channels` (training trials only)."""
    cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    subset = subset_band_covariances(band_covs, channels)
    # cross_val_score indexes the estimator's X; wrap the band list in a helper
    # array of trial indices so each fold slices every band consistently.
    idx = np.arange(subset[0].shape[0])
    scores = []
    for tr, te in cv.split(idx, y):
        fb = FBCSP(seed=seed).fit([c[tr] for c in subset], y[tr])
        scores.append(fb.score([c[te] for c in subset], y[te]))
    return float(np.mean(scores))


def fb_greedy_forward(band_covs, y, max_size, seed=SEED):
    """Greedy forward selection, FBCSP scored. Mirrors select_channels.greedy_forward.

    `band_covs` is a list of (trials, channels, channels) arrays, one per band.
    Ties go to the lowest channel index, so the result is deterministic.
    """
    n_channels = band_covs[0].shape[1]
    selected, path = [], []
    remaining = list(range(n_channels))
    for _ in range(max_size):
        scores = [fb_subset_accuracy(band_covs, y, selected + [ch], seed) for ch in remaining]
        best = int(np.argmax(scores))
        selected.append(remaining.pop(best))
        path.append(scores[best])
    return selected, np.array(path)


def fb_test_accuracy(band_covs_train, y_train, band_covs_test, y_test, channels, seed=SEED):
    """Fit FBCSP on the training session's `channels`, score once on the test session."""
    tr = subset_band_covariances(band_covs_train, channels)
    te = subset_band_covariances(band_covs_test, channels)
    fb = FBCSP(seed=seed).fit(tr, y_train)
    return fb.score(te, y_test)
