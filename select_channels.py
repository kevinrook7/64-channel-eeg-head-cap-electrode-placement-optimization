"""Greedy forward channel selection with nested cross-validation (Phase 4).

Outer loop (5 folds): estimates how well a selected subset works on unseen trials.
Inner loop (5 folds, training trials only): scores candidate channels for greedy selection.
The outer test trials are never seen during selection.
"""

import time

import mne
import numpy as np
from joblib import Parallel, delayed  # installed with scikit-learn
from sklearn.model_selection import StratifiedKFold, cross_val_score

from baseline import N_CSP, N_SPLITS, SEED, make_csp_decoder, task_window
from preprocess import prepare_epochs

mne.set_log_level("WARNING")  # CSP otherwise prints for every fit

SUBSET_SIZES = [8, 16, 32]
INNER_SPLITS = 5
# Objective: score(S) = inner_accuracy(S) - LAMBDA * |S|
# LAMBDA = 0.005 means one extra channel must buy at least 0.5 accuracy points.
LAMBDA = 0.005
N_JOBS = -1  # use all CPU cores for candidate scoring


def csp_decoder_for(n_channels, seed=SEED):
    # CSP can't return more filters than channels.
    return make_csp_decoder(seed, n_components=min(N_CSP, n_channels))


def subset_accuracy(X, y, channels, seed=SEED):
    """Inner-CV accuracy of CSP + LR using only `channels` (training trials only)."""
    # Runs inside joblib worker processes, which don't inherit the log level set above.
    mne.set_log_level("WARNING")
    cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    decoder = csp_decoder_for(len(channels), seed)
    return cross_val_score(decoder, X[:, channels, :], y, cv=cv).mean()


def greedy_forward(X, y, max_size, seed=SEED, parallel=None):
    """Add one channel at a time, always the one that most improves inner-CV accuracy.

    Returns the selection order and the inner-CV accuracy after each addition.
    Ties go to the lowest channel index, so the result is deterministic.
    """
    selected, path = [], []
    remaining = list(range(X.shape[1]))
    for _ in range(max_size):
        scores = parallel(
            delayed(subset_accuracy)(X, y, selected + [ch], seed) for ch in remaining
        )
        best = int(np.argmax(scores))
        selected.append(remaining.pop(best))
        path.append(scores[best])
    return selected, np.array(path)


def penalized_size(path, lam=LAMBDA):
    """Subset size that maximizes inner_accuracy - lam * size along the greedy path."""
    sizes = np.arange(1, len(path) + 1)
    return int(sizes[np.argmax(path - lam * sizes)])


def test_accuracy(X_train, y_train, X_test, y_test, channels, seed=SEED):
    decoder = csp_decoder_for(len(channels), seed)
    decoder.fit(X_train[:, channels, :], y_train)
    return decoder.score(X_test[:, channels, :], y_test)


def nested_cv(X, y, seed=SEED):
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    all_channels = list(range(X.shape[1]))
    max_size = max(SUBSET_SIZES)
    results = {"full": [], "penalized": [], **{k: [] for k in SUBSET_SIZES}}
    selections, penalized_sizes = [], []

    with Parallel(n_jobs=N_JOBS) as parallel:
        for fold, (train, test) in enumerate(outer.split(X, y), start=1):
            start = time.time()
            X_tr, y_tr, X_te, y_te = X[train], y[train], X[test], y[test]

            # Selection sees training trials only.
            selected, path = greedy_forward(X_tr, y_tr, max_size, seed, parallel)
            k_pen = penalized_size(path)

            results["full"].append(test_accuracy(X_tr, y_tr, X_te, y_te, all_channels, seed))
            for k in SUBSET_SIZES:
                results[k].append(test_accuracy(X_tr, y_tr, X_te, y_te, selected[:k], seed))
            results["penalized"].append(
                test_accuracy(X_tr, y_tr, X_te, y_te, selected[:k_pen], seed)
            )
            selections.append(selected)
            penalized_sizes.append(k_pen)

            tests = ", ".join(f"{k}ch {results[k][-1]:.3f}" for k in SUBSET_SIZES)
            print(
                f"fold {fold}: full {results['full'][-1]:.3f} | {tests} | "
                f"penalized {k_pen}ch {results['penalized'][-1]:.3f} "
                f"({time.time() - start:.0f}s)"
            )
    return results, selections, penalized_sizes


def summarize(scores):
    scores = np.asarray(scores)
    return f"{scores.mean():.3f} +/- {scores.std(ddof=1):.3f}"


if __name__ == "__main__":
    epochs = prepare_epochs(1)
    X = task_window(epochs)
    y = epochs.events[:, 2]
    names = np.array(epochs.ch_names)
    majority = max((y == 2).sum(), (y == 3).sum()) / len(y)
    print("X shape (trials, channels, times):", X.shape)
    print(f"majority-class accuracy: {majority:.3f}\n")

    results, selections, penalized_sizes = nested_cv(X, y)

    print("\nOuter-fold test accuracy (subject 1, within-session, CSP + LR):")
    print(f"  full ({X.shape[1]}ch): {summarize(results['full'])}")
    for k in SUBSET_SIZES:
        print(f"  greedy {k}ch: {summarize(results[k])}")
    print(
        f"  penalized (lambda={LAMBDA}, sizes {penalized_sizes}): "
        f"{summarize(results['penalized'])}"
    )

    # Stability: does each fold pick the same channels?
    print("\nFirst 8 channels picked in each fold (in selection order):")
    for fold, selected in enumerate(selections, start=1):
        print(f"  fold {fold}: {', '.join(names[selected[:8]])}")
    counts = {}
    for selected in selections:
        for ch in names[selected[:8]]:
            counts[ch] = counts.get(ch, 0) + 1
    common = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print("Picked in the 8-channel subset (folds out of 5):")
    print("  " + ", ".join(f"{ch} {n}" for ch, n in common))
