"""Causal sliding-window decoding.

v1 decoded one fixed cue-locked window (0.5-4.0 s). Online control cannot do
that: it must emit a decision while the trial is still running, from whatever
post-cue data has arrived so far. This module turns each trial into a stream of
causal windows and decodes each one, so accuracy can be read as a function of
decision latency (how long after the cue the decision is made).

Causality. A window ending at time t contains only samples in (t - L, t], where
L is the window length. It never reads a sample after t. "Latency" is the window
END time relative to cue onset: latency = 1.5 s means the decision used the first
1.5 s of post-cue EEG and nothing later.

Decoder. The same CSP+LR used in v1 (`fast_csp.CovCSP` + logistic regression),
fit on windows cut the SAME way from the training session, so train and test
windows match. Covariances are per-window (channels x channels), so a channel
subset is a slice -- selection stays cheap and identical to v1's math.

No leakage: CSP filters and the classifier are fit on training-session windows
only; the stream is scored, never fitted on.
"""

import numpy as np

from fast_csp import CovCSP, make_cov_csp_decoder, subset_covariances


def window_bounds(sfreq, win_len, latencies, cue_offset):
    """Sample index ranges for causal windows ending at each latency.

    sfreq      : samples per second.
    win_len    : window length in seconds (the look-back span).
    latencies  : iterable of window END times, in seconds AFTER cue onset.
    cue_offset : seconds from the start of the epoch array to cue onset. For the
                 BNCI epochs here (tmin = -0.5) the cue sits 0.5 s in, so
                 cue_offset = 0.5.

    Returns a list of (start, stop) sample indices, one per latency. A window is
    skipped (returned as None) if it would start before the epoch's first sample.
    """
    bounds = []
    for lat in latencies:
        stop = int(round((cue_offset + lat) * sfreq))
        start = int(round((cue_offset + lat - win_len) * sfreq))
        bounds.append((start, stop) if start >= 0 else None)
    return bounds


def window_covariances(X, bounds):
    """Per-trial covariance of each causal window.

    X      : (trials, channels, times) epoch array.
    bounds : output of window_bounds(); one (start, stop) per latency.

    Returns a dict {latency_index -> covs (trials, channels, channels)} for the
    windows that fit inside the epoch. A window covariance is X_w X_w^T / T_w,
    the same uncentered estimator fast_csp.trial_covariances uses, so the online
    decoder and the v1 decoder see identical covariance math.
    """
    n_times = X.shape[2]
    out = {}
    for i, b in enumerate(bounds):
        if b is None:
            continue
        start, stop = b
        stop = min(stop, n_times)
        if stop - start < 2:            # need at least 2 samples for a covariance
            continue
        Xw = X[:, :, start:stop]
        out[i] = np.einsum("nct,ndt->ncd", Xw, Xw) / Xw.shape[2]
    return out


def fit_window_decoder(covs_train, y_train, channels=None, seed=0, n_components=6):
    """Fit CSP+LR on training-session window covariances (optionally a channel subset)."""
    if channels is not None:
        covs_train = subset_covariances(covs_train, channels)
    n = min(n_components, covs_train.shape[1])
    decoder = make_cov_csp_decoder(seed, n_components=n)
    decoder.fit(covs_train, y_train)
    return decoder


def score_window(decoder, covs_test, y_test, channels=None):
    """Accuracy of a fitted decoder on one window's test covariances."""
    if channels is not None:
        covs_test = subset_covariances(covs_test, channels)
    return decoder.score(covs_test, y_test)


def latency_curve(X_train, y_train, X_test, y_test, sfreq, latencies,
                  win_len=2.0, cue_offset=0.5, channels=None, seed=0):
    """Accuracy vs decision latency for one train/test session pair.

    For each latency the window decoder is fit on the TRAINING session's window
    (ending at that latency) and scored on the TEST session's matching window.
    Train and test windows are cut identically, so the comparison is fair.

    Returns (kept_latencies, accuracies) as parallel arrays -- only latencies
    whose window fits inside both epochs are returned.
    """
    bounds = window_bounds(sfreq, win_len, latencies, cue_offset)
    covs_tr = window_covariances(X_train, bounds)
    covs_te = window_covariances(X_test, bounds)
    lats, accs = [], []
    for i, lat in enumerate(latencies):
        if i not in covs_tr or i not in covs_te:
            continue
        decoder = fit_window_decoder(covs_tr[i], y_train, channels, seed)
        accs.append(score_window(decoder, covs_te[i], y_test, channels))
        lats.append(lat)
    return np.array(lats), np.array(accs)


def crosses_precue(sfreq, win_len, latency, cue_offset):
    """True if the causal window ending at `latency` dips before the cue.

    A window that starts before the cue mixes resting EEG into the covariance, so
    its accuracy is not a clean "imagery decoded by time t" number. The sweep uses
    this to flag (and optionally drop) such points rather than plotting them as if
    they were pure post-cue decisions.
    """
    start = int(round((cue_offset + latency - win_len) * sfreq))
    return start < int(round(cue_offset * sfreq))


def window_length_sweep(X_train, y_train, X_test, y_test, sfreq, win_lens, latencies,
                        cue_offset=0.5, channels=None, seed=0, drop_precue=True):
    """Accuracy vs decision latency for several window LENGTHS.

    Returns {win_len -> (latencies, accuracies)}. When drop_precue is True, a
    latency whose window crosses the pre-cue baseline is skipped for that window
    length, so every plotted point is a clean post-cue decision. This is the sweep
    that separates "longer window = more signal" from "earlier window = less data
    but also, at short latency, contaminated by pre-cue baseline".
    """
    out = {}
    for wl in win_lens:
        keep = [lat for lat in latencies
                if not (drop_precue and crosses_precue(sfreq, wl, lat, cue_offset))]
        lats, accs = latency_curve(X_train, y_train, X_test, y_test, sfreq, keep,
                                   win_len=wl, cue_offset=cue_offset,
                                   channels=channels, seed=seed)
        out[wl] = (lats.tolist(), accs.tolist())
    return out
