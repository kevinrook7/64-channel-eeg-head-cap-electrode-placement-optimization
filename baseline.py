"""All-channel baseline for left vs right motor imagery (Phase 3)."""

import numpy as np
from mne.decoding import CSP
from preprocess import prepare_epochs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASK_TMIN = 0.5
TASK_TMAX = 4.0
SEED = 0
N_SPLITS = 5
N_CSP = 6  # CSP filters kept: 3 "loud for left" + 3 "loud for right"

def log_band_power(epochs, tmin=TASK_TMIN, tmax=TASK_TMAX):
    task = epochs.copy().crop(tmin=tmin, tmax=tmax)
    x = task.get_data()
    power = np.mean(x **2, axis=2)
    return np.log(power)

def make_decoder(seed=SEED):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000,random_state=seed)),
    ])

def task_window(epochs, tmin=TASK_TMIN, tmax=TASK_TMAX):
    """Raw epoch data (trials, channels, times) for the imagery window."""
    return epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()

def make_csp_decoder(seed=SEED, n_components=N_CSP):
    # CSP uses the labels, so it sits inside the pipeline and is refit on
    # each training fold only. log=True returns log(variance) per filter.
    return Pipeline([
        ("csp", CSP(n_components=n_components, log=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
    ])

def cross_validate_accuracy(X, y, decoder=None, seed=SEED, n_splits=N_SPLITS):
    # Same seed -> same folds for every decoder, so scores are comparable.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if decoder is None:
        decoder = make_decoder(seed)
    return cross_val_score(decoder, X, y, cv=cv, scoring="accuracy")

if __name__ == "__main__":
    epochs = prepare_epochs(1)
    X = log_band_power(epochs)
    y = epochs.events[:, 2] 
    print("X shape (trials, channels):", X.shape)
    print("left trials:", (y == 2).sum(), " right trials:", (y == 3).sum())
    print("trial 0, first 5 channels:", np.round(X[0, :5], 2))
    scores = cross_validate_accuracy(X, y)
    print("fold accuracies:", np.round(scores, 3))
    print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std(ddof=1):.3f}")
    n_left = (y == 2).sum()
    n_right = (y == 3).sum()
    majority = max(n_left, n_right) / len(y)
    print(f"majority-class accuracy: {majority:.3f}")
    print(f"model minus majority: {scores.mean() - majority:+.3f}")

    # CSP + logistic regression on the same trials and folds.
    X_csp = task_window(epochs)
    print("\nCSP input shape (trials, channels, times):", X_csp.shape)
    csp_scores = cross_validate_accuracy(X_csp, y, decoder=make_csp_decoder())
    print("CSP fold accuracies:", np.round(csp_scores, 3))
    print(f"CSP CV accuracy: {csp_scores.mean():.3f} +/- {csp_scores.std(ddof=1):.3f}")
    print(f"CSP minus majority: {csp_scores.mean() - majority:+.3f}")
    print(f"CSP minus band power: {csp_scores.mean() - scores.mean():+.3f}")
