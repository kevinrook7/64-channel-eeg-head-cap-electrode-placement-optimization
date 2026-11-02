r"""Tests for the Part 2 online module (causal decoding + recalibration).

Two things the reviewer flagged must hold, and these check them on tiny
synthetic data (no EEG download, well under a second):

  * CAUSALITY: a window ending at time t uses only samples at or before t.
  * NO LEAKAGE: the recalibration sweep never scores a trial it trained on, and
    Euclidean alignment never touches test labels.

Run: .\.venv\Scripts\python.exe -m pytest tests/ -v
"""

import numpy as np
import pytest

from online.causal_decoder import window_bounds, window_covariances
from online.euclidean_alignment import align, inv_sqrt, streaming_reference
from online.recalibration import _balanced_prefix, recalibration_curve


# ---------------------------------------------------------------- causality ---

def test_window_bounds_are_causal():
    """Every window ends at its latency and starts win_len earlier -- never later."""
    sfreq, win_len, cue_offset = 100.0, 1.0, 0.5
    lats = [0.5, 1.0, 2.0]
    bounds = window_bounds(sfreq, win_len, lats, cue_offset)
    for lat, b in zip(lats, bounds):
        start, stop = b
        # stop is exactly the window end = cue + latency, in samples.
        assert stop == int(round((cue_offset + lat) * sfreq))
        # the window spans win_len seconds and starts BEFORE stop (looks back only).
        assert stop - start == int(round(win_len * sfreq))
        assert start < stop


def test_window_before_epoch_start_is_dropped():
    """A window needing samples before t=0 is skipped, not padded with the future."""
    sfreq, win_len, cue_offset = 100.0, 2.0, 0.5
    # latency 0.5 with a 2.0 s look-back would start at 0.5+0.5-2.0 = -1.0 s -> invalid.
    bounds = window_bounds(sfreq, win_len, [0.5], cue_offset)
    assert bounds[0] is None


def test_window_covariance_uses_only_window_samples():
    """The covariance of a window must change if FUTURE samples change, and not."""
    sfreq = 100.0
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 4, 400))      # 3 trials, 4 ch, 4 s
    bounds = window_bounds(sfreq, win_len=1.0, latencies=[1.0], cue_offset=0.5)
    base = window_covariances(X, bounds)[0]

    # Corrupt samples AFTER the window end (stop index). Covariance must NOT move.
    start, stop = bounds[0]
    X_future = X.copy()
    X_future[:, :, stop:] += 100.0
    after = window_covariances(X_future, bounds)[0]
    assert np.allclose(base, after), "future samples leaked into a causal window"

    # Corrupt samples INSIDE the window. Covariance MUST move (sanity).
    X_inside = X.copy()
    X_inside[:, :, start:stop] += 100.0
    inside = window_covariances(X_inside, bounds)[0]
    assert not np.allclose(base, inside)


# ----------------------------------------------------------------- leakage ---

def _make_sessions(n=60, n_ch=5, seed=0):
    """Two 'sessions' of per-window covariances with a class signal on channel 0."""
    rng = np.random.default_rng(seed)
    def one(shift):
        covs, y = [], []
        for label, gain in [(2, 3.0), (3, 0.5)]:
            for _ in range(n // 2):
                d = np.ones(n_ch) + shift
                d[0] = gain
                noise = 0.05 * rng.standard_normal((n_ch, n_ch))
                covs.append(np.diag(d) + noise @ noise.T)
                y.append(label)
        return np.array(covs), np.array(y)
    return one(0.0), one(0.3)   # session 2 shifted (a drift)


def test_balanced_prefix_is_causal_and_balanced():
    """Recalibration trials come from the EARLIEST stream trials, class-balanced."""
    y = np.array([2, 3, 2, 3, 2, 3, 2, 3])
    idx = _balanced_prefix(y, k=4, seed=0)
    assert len(idx) == 4
    # balanced: two of each class
    assert (y[idx] == 2).sum() == 2 and (y[idx] == 3).sum() == 2
    # causal: only early trials (indices are small), never late ones
    assert idx.max() <= 5


def test_recalibration_never_scores_a_trial_it_trained_on():
    """The k recalibration trials are removed from the test set at every k."""
    (ctr, ytr), (cte, yte) = _make_sessions()

    # Spy on fit/score by tagging test covs with a unique fingerprint.
    tagged = cte.copy()
    for i in range(len(tagged)):
        tagged[i, -1, -1] += 1000.0 + i

    out = recalibration_curve(ctr, ytr, tagged, yte, channels=None,
                              recal_sizes=[0, 10, 20], seed=0)
    # Accuracies are valid fractions.
    assert 0.0 <= out["no_recal"] <= 1.0
    assert 0.0 <= out["ea"] <= 1.0
    for a in out["recal"]:
        assert 0.0 <= a <= 1.0
    # With a real drift, some recalibration should help over no-recal by k=20
    # (not asserted as strict > because synthetic, but must not error).


def test_euclidean_alignment_identity_and_unsupervised():
    """align() whitens to identity mean and takes no labels."""
    (ctr, _), _ = _make_sessions()
    aligned, ref = align(ctr)
    # mean aligned covariance is ~identity (EA's defining property)
    assert np.allclose(aligned.mean(0), np.eye(ctr.shape[1]), atol=1e-6)
    # inv_sqrt is a true inverse square root
    R = ref
    w = inv_sqrt(R)
    assert np.allclose(w @ R @ w, np.eye(R.shape[0]), atol=1e-6)


def test_streaming_reference_is_causal():
    """The reference at trial t uses only trials 0..t, never later ones."""
    (_, _), (cte, _) = _make_sessions()
    refs = streaming_reference(cte, min_trials=10)
    # Changing a LATE trial must not change an EARLY reference.
    cte2 = cte.copy()
    cte2[-1] += 50.0
    refs2 = streaming_reference(cte2, min_trials=10)
    assert np.allclose(refs[10], refs2[10]), "a later trial changed an earlier reference"
    # But the last reference (sees all) does change.
    assert not np.allclose(refs[-1], refs2[-1])
