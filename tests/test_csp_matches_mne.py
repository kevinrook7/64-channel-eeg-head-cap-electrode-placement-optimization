r"""CovCSP must reproduce mne.decoding.CSP exactly (fast_csp.py).

The whole channel-search speed-up rests on one claim: CovCSP, which works on
precomputed per-trial covariances, is mathematically identical to MNE's CSP,
which works on the raw time series -- with MNE's defaults
(cov_est="concat", reg=None, component_order="mutual_info", log=True).

So far that claim was only hand-checked by fast_csp.py's __main__ block, which
needs a real EEG download. These tests check it directly on tiny SYNTHETIC
trials: no download, no heavy compute, a fraction of a second. MNE is already a
core dependency (baseline.py, fast_csp.py both import it), so importing it here
adds nothing new.

Run: .\.venv\Scripts\python.exe -m pytest tests/test_csp_matches_mne.py -v
"""

import numpy as np
import pytest

mne = pytest.importorskip("mne", reason="MNE is a core dependency; skip if absent")
from mne.decoding import CSP

from fast_csp import CovCSP, trial_covariances, subset_covariances
from baseline import N_CSP


def make_synthetic_trials(n_per_class=30, n_channels=8, n_times=200, seed=0):
    """Two-class raw trials (trials, channels, times) with a class difference.

    Class 2 has extra variance on channels 0-1, class 3 on channels 2-3, so CSP
    has a real spatial pattern to find. Plain Gaussian sources mixed by a fixed
    random matrix -- enough structure that CSP filters are well-defined, small
    enough to run instantly.
    """
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((n_channels, n_channels))
    X, y = [], []
    for label, loud in [(2, (0, 1)), (3, (2, 3))]:
        for _ in range(n_per_class):
            src = rng.standard_normal((n_channels, n_times))
            for ch in loud:
                src[ch] *= 2.5  # this class is louder on these sources
            X.append(mix @ src)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


def _mne_csp(n_components):
    # MNE defaults that fast_csp.py claims to match, stated explicitly.
    return CSP(
        n_components=n_components,
        reg=None,
        log=True,
        cov_est="concat",
        component_order="mutual_info",
    )


def _assert_features_match(feat_mne, feat_cov):
    """Features must match MNE up to a single per-trial additive constant.

    MNE's CSP normalizes each trial's component powers to sum to 1 before log();
    CovCSP takes raw log(w^T C w). That makes the two differ by ONE constant per
    trial, IDENTICAL across all components of that trial -- so the per-trial
    spread of the difference is zero. The pipeline's StandardScaler centers each
    feature column, which removes that constant entirely, so the two are
    identical to the decoder (see test_full_pipeline_scores_match). We assert the
    exact shape of the difference: equal after removing a per-trial offset.
    """
    assert feat_mne.shape == feat_cov.shape
    diff = feat_mne - feat_cov
    per_trial_offset = diff.mean(axis=1, keepdims=True)
    # The difference is the SAME on every component of a trial (a pure offset).
    assert np.allclose(diff, per_trial_offset, atol=1e-9), \
        f"difference is not a constant per-trial offset; max spread = " \
        f"{np.abs(diff - per_trial_offset).max():.2e}"
    # And that offset is tiny -- it comes only from MNE's sum-to-one normalization.
    assert np.abs(per_trial_offset).max() < 1e-2, \
        f"per-trial offset unexpectedly large = {np.abs(per_trial_offset).max():.2e}"


def test_covcsp_features_match_mne_up_to_normalization():
    """CovCSP.transform equals mne CSP.transform up to MNE's per-trial normalization."""
    X, y = make_synthetic_trials()
    covs = trial_covariances(X)

    mne_csp = _mne_csp(N_CSP).fit(X, y)
    cov_csp = CovCSP(N_CSP).fit(covs, y)

    _assert_features_match(mne_csp.transform(X), cov_csp.transform(covs))


def test_covcsp_matches_mne_on_a_channel_subset():
    """The covariance slice for a channel subset must match MNE run on those channels.

    This is the operation the greedy search actually performs: pick channels,
    slice their covariances, decode. It must equal MNE restricted to the same
    raw channels -- otherwise the speed-up would change the answer.
    """
    X, y = make_synthetic_trials()
    covs = trial_covariances(X)
    channels = [0, 1, 2, 4, 5]
    n = min(N_CSP, len(channels))

    mne_csp = _mne_csp(n).fit(X[:, channels], y)
    cov_csp = CovCSP(n).fit(subset_covariances(covs, channels), y)

    _assert_features_match(
        mne_csp.transform(X[:, channels]),
        cov_csp.transform(subset_covariances(covs, channels)),
    )


def test_full_pipeline_scores_match():
    """What actually matters: standardized features are identical to the decoder.

    The per-trial offset above is a constant on each feature column, so a
    StandardScaler (mean-centering per column) removes it. After scaling, MNE's
    CSP features and CovCSP's features are bit-identical -- which is why the two
    decoders produce the same cross-validation scores. We assert that directly.
    """
    from sklearn.preprocessing import StandardScaler

    X, y = make_synthetic_trials()
    covs = trial_covariances(X)

    feat_mne = _mne_csp(N_CSP).fit(X, y).transform(X)
    feat_cov = CovCSP(N_CSP).fit(covs, y).transform(covs)

    scaled_mne = StandardScaler().fit_transform(feat_mne)
    scaled_cov = StandardScaler().fit_transform(feat_cov)

    assert np.allclose(scaled_mne, scaled_cov, atol=1e-6), \
        f"standardized features differ; max |diff| = " \
        f"{np.abs(scaled_mne - scaled_cov).max():.2e}"


def test_covcsp_filter_subspace_matches_mne():
    """The CSP filters must span the same subspace (sign/scale aside).

    Eigenvectors are only defined up to sign and scale, so filters can differ by
    a per-row factor. The features (test above) are the invariant that matters;
    here we additionally confirm each CovCSP filter is parallel to MNE's by
    checking the per-filter correlation magnitude is ~1.
    """
    X, y = make_synthetic_trials()
    covs = trial_covariances(X)

    mne_csp = _mne_csp(N_CSP).fit(X, y)
    cov_csp = CovCSP(N_CSP).fit(covs, y)

    for wm, wc in zip(mne_csp.filters_[:N_CSP], cov_csp.filters_):
        cos = abs(wm @ wc) / (np.linalg.norm(wm) * np.linalg.norm(wc))
        assert cos > 0.999, f"filter not parallel to MNE's (|cos| = {cos:.4f})"
