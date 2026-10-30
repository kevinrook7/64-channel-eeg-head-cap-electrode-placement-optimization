r"""No-leakage tests for greedy channel selection (select_channels.py).

The research rests on one rule: channel selection and all model fitting use
TRAINING trials only; the held-out test trials are never seen during selection.
These tests check that rule directly, on tiny synthetic covariances so they run
in well under a second with no EEG download and no heavy compute.

Run: .\.venv\Scripts\python.exe -m pytest tests/ -v
"""

import numpy as np
import pytest

from select_channels import greedy_forward, nested_cv, subset_accuracy
from select_channels import test_accuracy as fit_and_score  # aliased: pytest must not collect it as a test


def make_separable_covs(n_per_class=40, n_channels=6, signal_channels=(0, 1), seed=0):
    """Synthetic per-trial covariances where only `signal_channels` carry the class.

    Class 2 has high variance on the signal channels, class 3 has low variance
    there; every other channel is identical noise. A good selector must pick the
    signal channels, and must do so without peeking at labels it is not given.
    Returns (covs [trials, ch, ch], y).
    """
    rng = np.random.default_rng(seed)
    covs, y = [], []
    for label, gain in [(2, 3.0), (3, 0.4)]:
        for _ in range(n_per_class):
            d = np.ones(n_channels)
            for ch in signal_channels:
                d[ch] = gain * (1.0 + 0.1 * rng.standard_normal())
            # small random symmetric noise so trials are not identical
            noise = 0.05 * rng.standard_normal((n_channels, n_channels))
            cov = np.diag(d) + noise @ noise.T
            covs.append(cov)
            y.append(label)
    covs = np.array(covs)
    y = np.array(y)
    order = rng.permutation(len(y))
    return covs[order], y[order]


def test_greedy_sees_only_the_covariances_it_is_handed():
    """greedy_forward must never index trials outside the array passed to it.

    We pass ONLY a training slice. If selection leaked test trials it would need
    indices beyond the slice length; every returned channel index must instead
    be a valid CHANNEL index of the training array, and the function must touch
    no trial outside it. We verify by handing it a train slice and confirming the
    picks are channel indices in range and the call does not mutate the inputs.
    """
    covs, y = make_separable_covs()
    n_train = 50
    c_tr, y_tr = covs[:n_train].copy(), y[:n_train].copy()
    c_tr_before = c_tr.copy()

    selected, path = greedy_forward(c_tr, y_tr, max_size=4, seed=0)

    # Picks are channel indices of the training array, not trial indices.
    assert all(0 <= ch < c_tr.shape[1] for ch in selected)
    assert len(selected) == 4
    assert len(set(selected)) == 4  # no channel chosen twice
    # Selection did not mutate the training data it was given.
    assert np.array_equal(c_tr, c_tr_before)
    # Inner-CV accuracy is a real fraction.
    assert np.all((path >= 0.0) & (path <= 1.0))


def test_greedy_finds_the_signal_channels():
    """On separable data, greedy's first picks must be the signal channels.

    This is the positive control: if greedy could NOT find channels 0 and 1 the
    leakage test would be meaningless (a selector that ignores labels would also
    'not leak'). Signal lives in channels 0 and 1.
    """
    covs, y = make_separable_covs(signal_channels=(0, 1))
    selected, _ = greedy_forward(covs, y, max_size=2, seed=0)
    assert set(selected) == {0, 1}


def test_train_and_test_covariances_are_disjoint_in_nested_cv():
    """The decoder is fit on train covs and scored on test covs that share no trial.

    We wrap test_accuracy so every call records the train/test arrays it was fit
    and scored on, then run nested_cv and assert no trial-row appears in both the
    train and test set of the same fold.
    """
    covs, y = make_separable_covs()
    # Tag each trial with a unique fingerprint on an unused matrix entry so we can
    # tell train rows from test rows after slicing.
    tagged = covs.copy()
    for i in range(len(tagged)):
        tagged[i, -1, -1] += 1000.0 + i  # unique, does not affect class signal

    seen = []
    import select_channels as sc
    original = sc.test_accuracy

    def spy(covs_train, y_train, covs_test, y_test, channels, seed=sc.SEED):
        train_tags = {float(c[-1, -1]) for c in covs_train}
        test_tags = {float(c[-1, -1]) for c in covs_test}
        seen.append((train_tags, test_tags))
        return original(covs_train, y_train, covs_test, y_test, channels, seed)

    sc.test_accuracy = spy
    try:
        nested_cv(tagged, y, seed=0, verbose=False, sizes=[2])
    finally:
        sc.test_accuracy = original

    assert seen, "nested_cv should have scored at least one fold"
    for train_tags, test_tags in seen:
        # No trial is in both the training and the test set of the same fold.
        assert train_tags.isdisjoint(test_tags)


def test_subset_accuracy_uses_only_given_channels():
    """subset_accuracy on the signal channels must beat it on noise-only channels."""
    covs, y = make_separable_covs(signal_channels=(0, 1))
    signal = subset_accuracy(covs, y, [0, 1], seed=0)
    noise = subset_accuracy(covs, y, [3, 4], seed=0)
    assert signal > noise
    assert signal > 0.6  # clearly above chance on separable data
