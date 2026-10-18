"""All-channel baseline for left vs right motor imagery (Phase 3)."""

import numpy as np
from preprocess import prepare_epochs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASK_TMIN = 0.5
TASK_TMAX = 4.0
SEED = 0
N_SPLITS = 5

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

def cross_validate_accuracy(X, y, seed=SEED, n_splits=N_SPLITS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_score(make_decoder(), X, y, cv=cv, scoring="accuracy")

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
