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
  * EA's test reference is CAUSAL: trial t is whitened by the running mean of
    test covariances 0..t only (streaming_reference), never future trials. The
    headline "ea" number uses this causal reference. A transductive variant
    (reference from all test trials) is reported separately as "ea_transductive"
    for contrast only -- it is label-free but has seen the future, so it is not
    online and is never the headline.

Latency: the window length and decision latency are fixed for the sweep (default
a 2.0 s window ending 2.0 s after the cue) so the x-axis is purely "recalibration
trials". causal_decoder.latency_curve covers the orthogonal latency sweep.
"""

import numpy as np

from fast_csp import make_cov_csp_decoder, subset_covariances
from online.causal_decoder import window_bounds, window_covariances
from online.euclidean_alignment import align, reference_covariance

RECAL_SIZES = [0, 5, 10, 20, 40]
# Causal EA needs a few unlabeled trials before its running reference is stable;
# earlier trials reuse the first stable reference (see streaming_reference).
EA_MIN_TRIALS = 10
# Quick recalibration is averaged over this many random balanced draws of which
# k trials are used, so the reported curve is not one noisy draw.
N_DRAWS = 10


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


def _balanced_draw(y, k, rng):
    """Random class-balanced sample of k trial indices (for averaging over draws).

    Unlike _balanced_prefix (earliest k, for the strictly-causal demo), this draws
    k/2 of each class at random, so averaging over draws removes the which-trials
    sampling noise. Still label-aware (recalibration is supervised by definition).
    """
    classes = np.unique(y)
    per = k // len(classes)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        idx.extend(rng.choice(c_idx, size=min(per, len(c_idx)), replace=False).tolist())
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

    Deployment-realistic, fully causal scoring (reviewer round 3):
      * The k recalibration trials are the FIRST k trials of session 2 in stream
        order (not a random draw from anywhere) — a warm-up at the START of the
        day, never a trial from later than the trials it is scored against.
      * EVERY condition at EVERY k is scored on ONE fixed held-out set: the trials
        after the largest k (trials max(k)+1 .. end). So no_recal / EA / recal /
        EA+recal all see the identical test trials at every budget — the fair
        comparison the reviewer asked for, and it matches "calibrate at day start,
        then decode".
      * Causal EA's running reference still starts at trial 0, so by the time the
        fixed scoring set begins it has seen max(k) unlabeled trials — fair, and
        stated in the README.
    The first-k warm-up is a single deterministic draw, so the 10-random-draws
    averaging is dropped; a random-draw sensitivity check is kept under the
    '*_randdraw' keys for contrast only.
    """
    out = {"sizes": list(recal_sizes), "recal": []}

    # --- baseline: session-1 decoder, no recalibration (constant over k) ---
    base = fit_decoder(covs_tr, y_tr, channels, seed)
    out["no_recal"] = float(score(base, covs_te, y_te, channels))

    # --- Euclidean alignment ---
    # Training session is offline, so its reference is its own mean (fine).
    # The TEST session is a stream: trial t may use ONLY trials 0..t to form its
    # reference -- a causal running mean -- never future trials. The older
    # transductive variant (reference from ALL test trials) is kept for contrast
    # under "ea_transductive" so the gap between causal and transductive is visible.
    from online.euclidean_alignment import inv_sqrt, streaming_reference

    sub_tr = subset_covariances(covs_tr, channels) if channels is not None else covs_tr
    sub_te = subset_covariances(covs_te, channels) if channels is not None else covs_te
    n = min(6, sub_tr.shape[1])

    aligned_tr, _ = align(sub_tr)                       # train reference = session-1 mean
    ea_dec = make_cov_csp_decoder(seed, n_components=n)
    ea_dec.fit(aligned_tr, y_tr)

    # Causal EA: whiten each test trial by the running mean of trials seen so far.
    refs = streaming_reference(sub_te, min_trials=EA_MIN_TRIALS)   # (trials, c, c), causal
    aligned_causal = np.stack([inv_sqrt(refs[t]) @ sub_te[t] @ inv_sqrt(refs[t])
                               for t in range(len(sub_te))])
    preds = ea_dec.predict(aligned_causal)
    out["ea"] = float((preds == y_te).mean())           # headline EA is now CAUSAL

    # EA accuracy vs number of unlabeled trials seen, two views:
    #  - cumulative_acc: running mean over trials 0..t (what it was); kept for back-compat.
    #  - rolling_acc: accuracy in a sliding window of the last ROLL trials, which
    #    separates "the reference is stabilizing" from "these particular trials were
    #    easier/harder". A cumulative curve that peaks mid-stream is usually trial
    #    difficulty, not the reference; the rolling curve shows that directly.
    correct = (preds == y_te).astype(float)
    base_correct = (base.predict(sub_te) == y_te).astype(float)   # no-recal, same trials
    seen = np.arange(1, len(y_te) + 1)
    ROLL = 20
    rolling = np.full(len(correct), np.nan)
    rolling_base = np.full(len(correct), np.nan)
    for t in range(len(correct)):
        lo = max(0, t - ROLL + 1)
        rolling[t] = correct[lo:t + 1].mean()
        rolling_base[t] = base_correct[lo:t + 1].mean()
    out["ea_vs_seen"] = {
        "trials_seen": seen.tolist(),
        "cumulative_acc": np.cumsum(correct).tolist(),   # divide by trials_seen for running acc
        "rolling_acc": rolling.tolist(),                 # causal EA, rolling window
        "rolling_acc_norecal": rolling_base.tolist(),    # no-recal, SAME trials & window
        "rolling_diff": (rolling - rolling_base).tolist(),  # EA minus no-recal (reviewer pt 2)
        "rolling_window": ROLL,
    }

    # Transductive EA (reference from ALL test trials) -- NOT causal, for contrast only.
    aligned_trans, _ = align(sub_te)
    out["ea_transductive"] = float(ea_dec.score(aligned_trans, y_te))

    # --- quick recalibration, deployment-realistic (first-k warm-up, fixed test set) ---
    # Warm-up = first k trials in STREAM ORDER. Scoring set = the trials after the
    # LARGEST k, fixed across all conditions and all k, so every number below is on
    # the identical held-out trials. base_preds / preds are per-trial, so no_recal
    # and causal-EA are scored by masking the same fixed set.
    base_preds = base.predict(subset_covariances(covs_te, channels) if channels is not None else covs_te)
    k_max = max(recal_sizes)
    fixed = np.arange(k_max, len(y_te))                 # trials k_max .. end (e.g. 40..143)
    out["scored_trials"] = int(len(fixed))
    out["scored_from_trial"] = int(k_max)

    out["recal"], out["recal_ea"] = [], []              # plain recal / EA+recal combo
    out["no_recal_matched"], out["ea_matched"] = [], []  # same fixed set, for comparison
    for k in recal_sizes:
        # no-recal and causal-EA do not use labels; score them on the fixed set.
        out["no_recal_matched"].append(float((base_preds[fixed] == y_te[fixed]).mean()))
        out["ea_matched"].append(float((preds[fixed] == y_te[fixed]).mean()))
        if k == 0:
            out["recal"].append(out["no_recal_matched"][-1])
            out["recal_ea"].append(out["ea_matched"][-1])
            continue
        warm = np.arange(0, k)                           # first k trials, stream order
        # plain recalibration: session 1 + first-k labeled trials, refit, score on fixed set.
        covs_aug = np.concatenate([covs_tr, covs_te[warm]])
        y_aug = np.concatenate([y_tr, y_te[warm]])
        dec = fit_decoder(covs_aug, y_aug, channels, seed)
        out["recal"].append(float(score(dec, covs_te[fixed], y_te[fixed], channels)))
        # EA + recalibration: fit on aligned session 1 + aligned first-k, score on fixed set.
        ea_covs_aug = np.concatenate([aligned_tr, aligned_causal[warm]])
        ea_dec_k = make_cov_csp_decoder(seed, n_components=n)
        ea_dec_k.fit(ea_covs_aug, y_aug)
        out["recal_ea"].append(float((ea_dec_k.predict(aligned_causal[fixed]) == y_te[fixed]).mean()))

    # --- sensitivity check: random balanced warm-up draws (NOT causal; contrast only) ---
    # Kept so a reader can see the first-k result is not a lucky ordering. Scored on
    # the complement of each draw, averaged over N_DRAWS; not the headline.
    rd = []
    for k in recal_sizes:
        if k == 0:
            rd.append(out["no_recal"]); continue
        vals = []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(seed * 100 + d)
            ridx = _balanced_draw(y_te, k, rng)
            mask = np.ones(len(y_te), dtype=bool); mask[ridx] = False
            ca = np.concatenate([covs_tr, covs_te[ridx]]); ya = np.concatenate([y_tr, y_te[ridx]])
            dec = fit_decoder(ca, ya, channels, seed)
            vals.append(score(dec, covs_te[mask], y_te[mask], channels))
        rd.append(float(np.mean(vals)))
    out["recal_randdraw"] = rd
    return out


def window_covs_for(X, sfreq, win_len, latency, cue_offset=0.5):
    """Covariances of the single causal window used by the recalibration sweep."""
    bounds = window_bounds(sfreq, win_len, [latency], cue_offset)
    covs = window_covariances(X, bounds)
    if 0 not in covs:
        raise ValueError("window does not fit inside the epoch; check win_len/latency")
    return covs[0]
