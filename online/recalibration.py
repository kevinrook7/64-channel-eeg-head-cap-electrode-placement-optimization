"""Quick-recalibration sweep: can a session-1 channel set recover on session 2
with a few labeled trials?

Design (per subject, honest and causal):

  1. SELECT channels on session 1 only -- greedy forward selection, labels from
     session 1. Session 2 never informs the choice. (Set channels=None to use
     all 22 and measure recalibration alone.)
  2. FIT the base decoder (CSP+LR) on the whole of session 1, on the causal
     imagery window.
  3. Treat session 2 as an arriving stream. The FIRST k trials are the
     recalibration set (labeled, as a real user's short warm-up would be); the
     REMAINING trials are the held-out test set, scored once. k sweeps over
     0, 5, 10, 20, 40.

Three conditions compared at each k:

  * no_recal  : session-1 decoder applied as-is (the v1 cross-session number).
  * ea        : Euclidean alignment -- session-1 decoder, but covariances aligned
                by each session's reference. The test reference is built from the
                UNLABELED test stream, so EA spends zero labels (k is ignored for
                EA; it is drawn as a flat line for comparison).
  * recal     : the k labeled recalibration trials are ADDED to session 1 and the
                decoder is refit. This is the "short warm-up" the product would do.

Leakage controls:
  * The k recalibration trials are removed from the test set, so no trial is both
    trained on and scored.
  * Channel selection used session 1 only (step 1).
  * EA's test reference uses unlabeled covariances only.

Latency: the window length and decision latency are fixed for the sweep (default
a 2.0 s window ending 2.0 s after the cue) so the x-axis is purely "recalibration
trials". causal_decoder.latency_curve covers the orthogonal latency sweep.
"""

import numpy as np

from fast_csp import make_cov_csp_decoder, subset_covariances
from online.causal_decoder import window_bounds, window_covariances
from online.euclidean_alignment import align, reference_covariance

RECAL_SIZES = [0, 5, 10, 20, 40]


def _balanced_prefix(y, k, seed=0):
    """Indices of the first k stream trials, kept class-balanced.

    A real warm-up would collect roughly equal left/right cues. Taking the first
    k in order could be class-skewed, so we take the first k/2 of each class in
    arrival order -- still causal (only early trials), but balanced.
    """
    if k == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = k // len(classes)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0][:per]      # earliest `per` trials of class c
        idx.extend(c_idx.tolist())
    return np.array(sorted(idx))


def fit_decoder(covs, y, channels, seed, n_components=6):
    sub = subset_covariances(covs, channels) if channels is not None else covs
    n = min(n_components, sub.shape[1])
    dec = make_cov_csp_decoder(seed, n_components=n)
    dec.fit(sub, y)
    return dec


def score(dec, covs, y, channels):
    sub = subset_covariances(covs, channels) if channels is not None else covs
    return dec.score(sub, y)


def recalibration_curve(covs_tr, y_tr, covs_te, y_te, channels=None,
                        recal_sizes=RECAL_SIZES, seed=0):
    """Accuracy vs recalibration-trial count for one subject.

    covs_tr/covs_te : per-window covariances for session 1 / session 2
                      (trials, channels, channels), from the SAME causal window.
    Returns a dict with 'sizes', 'no_recal', 'ea', and 'recal' (list over sizes).
    """
    out = {"sizes": list(recal_sizes), "recal": []}

    # --- baseline: session-1 decoder, no recalibration (constant over k) ---
    base = fit_decoder(covs_tr, y_tr, channels, seed)
    out["no_recal"] = float(score(base, covs_te, y_te, channels))

    # --- Euclidean alignment: unsupervised transfer, zero labels (constant) ---
    sub_tr = subset_covariances(covs_tr, channels) if channels is not None else covs_tr
    sub_te = subset_covariances(covs_te, channels) if channels is not None else covs_te
    aligned_tr, _ = align(sub_tr)                       # train reference = session-1 mean
    aligned_te, _ = align(sub_te)                       # test reference  = unlabeled session-2 mean
    n = min(6, sub_tr.shape[1])
    ea_dec = make_cov_csp_decoder(seed, n_components=n)
    ea_dec.fit(aligned_tr, y_tr)
    out["ea"] = float(ea_dec.score(aligned_te, y_te))

    # --- quick recalibration: add k labeled test trials, refit, score the rest ---
    for k in recal_sizes:
        recal_idx = _balanced_prefix(y_te, k, seed)
        mask = np.ones(len(y_te), dtype=bool)
        mask[recal_idx] = False                          # held-out test = everything not used to recalibrate
        if k == 0:
            out["recal"].append(out["no_recal"])
            continue
        covs_aug = np.concatenate([covs_tr, covs_te[recal_idx]])
        y_aug = np.concatenate([y_tr, y_te[recal_idx]])
        dec = fit_decoder(covs_aug, y_aug, channels, seed)
        out["recal"].append(float(score(dec, covs_te[mask], y_te[mask], channels)))
    return out


def window_covs_for(X, sfreq, win_len, latency, cue_offset=0.5):
    """Covariances of the single causal window used by the recalibration sweep."""
    bounds = window_bounds(sfreq, win_len, [latency], cue_offset)
    covs = window_covariances(X, bounds)
    if 0 not in covs:
        raise ValueError("window does not fit inside the epoch; check win_len/latency")
    return covs[0]
