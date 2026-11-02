"""Euclidean Alignment (EA) -- a cheap, well-known session-transfer baseline.

He & Wu (2020), "Transfer Learning for Brain-Computer Interfaces: A Euclidean
Space Data Alignment Approach". The idea: each session's trials are whitened by
that session's own mean covariance, so both sessions land in a common space and
a decoder trained on session 1 transfers to session 2 without retraining.

    R      = mean over that session's trials of C_i        (reference covariance)
    R^-1/2 = inverse matrix square root of R
    C_i'   = R^-1/2 C_i R^-1/2                              aligned covariance

Why it is an honest transfer baseline here:
  * The reference R is computed from the session's covariances ONLY -- no labels.
    The test session's R uses the incoming (unlabeled) stream, which is exactly
    what a real device has: it sees signals, not answers.
  * After alignment each session's mean covariance is the identity, so the
    session-to-session shift (the drift v1 measured) is removed at first order
    without ever looking at a test label.

This sits beside quick recalibration: recalibration spends a few labeled trials;
EA spends none, using only the shape of the unlabeled stream. Comparing them
shows whether the small labeled budget actually buys anything over free
unsupervised alignment.
"""

import numpy as np
import scipy.linalg


def inv_sqrt(mat):
    """Symmetric inverse square root of a positive-definite matrix."""
    evals, evecs = scipy.linalg.eigh(mat)
    evals = np.clip(evals, 1e-12, None)        # guard tiny/negative eigenvalues
    return evecs @ np.diag(evals ** -0.5) @ evecs.T


def reference_covariance(covs):
    """Mean covariance of a set of per-trial covariances (the EA reference R)."""
    return covs.mean(axis=0)


def align(covs, reference=None):
    """Whiten covariances by a reference (their own mean if none is given).

    Returns (aligned_covs, reference). Pass the returned reference back in to
    align a later chunk of the SAME session with the same transform -- so a
    streaming reference estimated from the trials seen so far can be reused.
    """
    if reference is None:
        reference = reference_covariance(covs)
    whitener = inv_sqrt(reference)
    aligned = np.einsum("ij,njk,kl->nil", whitener, covs, whitener)
    return aligned, reference


def streaming_reference(covs, min_trials=10):
    """Running EA reference after each incoming trial (causal, unlabeled).

    A real stream does not have all test covariances up front; it accumulates
    them. This returns, for trial index t (t >= min_trials-1), the mean of
    covariances 0..t -- the reference a causal device could have formed by then.
    Earlier trials (fewer than min_trials seen) reuse the first stable reference.
    """
    running = np.zeros_like(covs[0])
    refs = []
    for t in range(len(covs)):
        running = running + covs[t]
        refs.append(running / (t + 1))
    first = max(min_trials - 1, 0)
    for t in range(first):
        refs[t] = refs[first]
    return np.stack(refs)
